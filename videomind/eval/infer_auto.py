# Copyright (c) 2025 Ye Liu. Licensed under the BSD-3-Clause License.
# ActiveLook: Feedback-Driven Multi-Scale Active Perception for Long Video Reasoning

import argparse
import copy
import json
from contextlib import nullcontext

import nncore
import torch

from videomind.constants import GROUNDER_PROMPT, PLANNER_PROMPT, VERIFIER_PROMPT
from videomind.dataset.hybrid import DATASETS
from videomind.dataset.utils import process_vision_info
from videomind.model.builder import build_model
from videomind.utils.io import get_duration, load_subtitle
from videomind.utils.parser import parse_query, parse_span


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset')
    parser.add_argument('--pred_path')
    parser.add_argument('--model_gnd_path')
    parser.add_argument('--model_ver_path')
    parser.add_argument('--model_pla_path')
    parser.add_argument('--model_ans_path')
    parser.add_argument('--split', default='test', choices=['train', 'valid', 'test'])
    parser.add_argument('--style', default='mcq', choices=['mcq', 'options', 'direct'])
    parser.add_argument('--use_subtitle', action='store_true')
    parser.add_argument('--auto_rephrasing', action='store_true')
    parser.add_argument('--auto_planning', action='store_true')
    parser.add_argument('--num_threads', type=int, default=1)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--chunk', type=int, default=1)
    parser.add_argument('--index', type=int, default=0)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    if args.chunk > 1:
        pred_path = nncore.join(args.pred_path, f'output_{args.index}.json')
    else:
        pred_path = nncore.join(args.pred_path, 'output.json')

    print(f'Dataset: {args.dataset}({args.split}) Chunk: {args.chunk} Index: {args.index} Output Path: {pred_path}')

    adapter_state = dict(planner=False, verifier=False, answerer=False)

    print('Initializing role *grounder*')
    model, processor = build_model(args.model_gnd_path, device=args.device)
    device = next(model.parameters()).device

    if args.model_pla_path is not None:
        adapter_path = nncore.join(args.model_pla_path, 'planner')
        if nncore.is_dir(adapter_path):
            print('Initializing role *planner*')
            model.load_adapter(adapter_path, adapter_name='planner')
            adapter_state['planner'] = True

    if args.model_ver_path is not None:
        adapter_path = nncore.join(args.model_ver_path, 'verifier')
        if nncore.is_dir(adapter_path):
            print('Initializing role *verifier*')
            model.load_adapter(adapter_path, adapter_name='verifier')
            adapter_state['verifier'] = True

    if args.model_ans_path is not None:
        adapter_path = nncore.join(args.model_ans_path, 'answerer')
        if nncore.is_dir(adapter_path):
            print('Initializing role *answerer*')
            model.load_adapter(adapter_path, adapter_name='answerer')
            adapter_state['answerer'] = True

    dataset_cls = DATASETS.get(args.dataset)

    annos = dataset_cls.load_annos(split=args.split)
    annos = [annos[i::args.chunk] for i in range(args.chunk)][args.index]

    dumps = []
    for i in nncore.ProgressBar(range(len(annos))):
        anno = copy.deepcopy(annos[i])
        dump = copy.deepcopy(annos[i])

        video_path, duration, span = anno['video_path'], anno.get('duration'), anno.get('span')

        if duration is None:
            duration = get_duration(video_path, num_threads=args.num_threads)
            dump['duration'] = duration

        print()
        print(video_path)
        print(duration)

        do_answering = all(k in anno for k in ('question', 'options'))

        if do_answering:
            question, options, ans = anno['question'], anno['options'], anno['ans']

            if args.style in ('mcq', 'options'):
                prompt = question + '\nOptions:'
                for idx, opt in enumerate(options):
                    # Safely handle option text
                    if opt and len(opt) > 0:
                        opt_text = opt[0].upper() + opt[1:]
                    else:
                        opt_text = '[empty]'
                        print(f'WARNING: Empty option at index {idx} for video {anno.get("video_path", "unknown")}')
                    prompt += f'\n({chr(ord("A") + idx)}) {opt_text}'
                prompt += '\nPlease only give the best option.'
            else:
                prompt = question

            print(prompt)
            print(options)
            print(ans)
        else:
            question = anno['query']
            print(question)

        do_grounding = True
        query = question
        dump['agents'] = []

        # ==================== Planner ====================
        if adapter_state['planner'] and (args.auto_rephrasing or args.auto_planning):
            print('=============== planner ===============')

            dump['agents'].append('planner')

            messages = [{
                'role': 'user',
                'content': [{
                    'type': 'video',
                    'video': video_path,
                    'num_threads': args.num_threads,
                    'min_pixels': 36 * 28 * 28,
                    'max_pixels': 64 * 28 * 28,
                    'max_frames': 100,
                    'fps': 1.0
                }, {
                    'type': 'text',
                    'text': PLANNER_PROMPT.format(question)
                }]
            }]

            text = processor.apply_chat_template(messages, add_generation_prompt=True)
            print(text)
            images, videos = process_vision_info(messages)
            data = processor(text=[text], images=images, videos=videos, return_tensors='pt')
            data = data.to(device)

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(device)

            model.base_model.disable_adapter_layers()
            model.base_model.enable_adapter_layers()
            model.set_adapter('planner')

            try:
                output_ids = model.generate(
                    **data,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                    repetition_penalty=None,
                    max_new_tokens=256)
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    torch.cuda.empty_cache()
                raise

            assert data.input_ids.size(0) == output_ids.size(0) == 1
            output_ids = output_ids[0, data.input_ids.size(1):]
            if output_ids[-1] == processor.tokenizer.eos_token_id:
                output_ids = output_ids[:-1]
            response = processor.decode(output_ids, clean_up_tokenization_spaces=False)
            print(response)

            dump['planner_response'] = response

            try:
                parsed = json.loads(response)
                action = parsed[0] if isinstance(parsed, list) else parsed
                if args.auto_rephrasing and action['type'].lower() == 'grounder' and action['value']:
                    query = action['value']
                    dump['planner_parsed_query'] = query
                elif args.auto_planning and action['type'].lower() == 'answerer':
                    do_grounding = False
            except Exception:
                print('WARNING: Failed to parse planner response')

        # ==================== Grounder (Multi-Scale Active Perception) ====================
        if do_grounding:
            print('=============== grounder (Multi-Scale) ===============')

            dump['agents'].append('grounder')

            query = parse_query(query)

            # Define multi-scale sampling configurations.
            # Each scale targets a different temporal/spatial trade-off:
            #   - standard:      baseline configuration for typical events
            #   - high_temporal: higher FPS and more frames for short transient events
            #   - high_spatial:  lower FPS with higher resolution for fine visual details
            scales = [
                {
                    'name': 'standard',
                    'fps': 1.0,
                    'max_frames': 150,
                    'min_pixels': 36 * 28 * 28,
                    'max_pixels': 64 * 28 * 28,
                    'description': 'Standard sampling (baseline)'
                },
                {
                    'name': 'high_temporal',
                    'fps': 1.5,
                    'max_frames': 200,
                    'min_pixels': 32 * 28 * 28,
                    'max_pixels': 56 * 28 * 28,
                    'description': 'Higher FPS, more frames'
                },
                {
                    'name': 'high_spatial',
                    'fps': 0.8,
                    'max_frames': 120,
                    'min_pixels': 48 * 28 * 28,
                    'max_pixels': 80 * 28 * 28,
                    'description': 'Lower FPS, higher resolution'
                }
            ]

            # Activate the grounder adapter once before starting the multi-scale loop.
            model.base_model.disable_adapter_layers()
            model.base_model.enable_adapter_layers()
            model.set_adapter('grounder')

            all_predictions = []
            all_confidences = []
            all_responses = []
            all_scale_names = []

            for round_idx, scale in enumerate(scales):
                print(f'\n--- Round {round_idx}: {scale["name"]} ---')
                print(f'   {scale["description"]}')
                print(f'   fps={scale["fps"]}, max_frames={scale["max_frames"]}, '
                      f'pixels={scale["min_pixels"]//784}~{scale["max_pixels"]//784}')

                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats(device)

                # Build scale-specific video inputs.
                messages = [{
                    'role': 'user',
                    'content': [{
                        'type': 'video',
                        'video': video_path,
                        'num_threads': args.num_threads,
                        'min_pixels': scale['min_pixels'],
                        'max_pixels': scale['max_pixels'],
                        'max_frames': scale['max_frames'],
                        'fps': scale['fps']
                    }, {
                        'type': 'text',
                        'text': GROUNDER_PROMPT.format(query)
                    }]
                }]

                text = processor.apply_chat_template(messages, add_generation_prompt=True)
                images, videos = process_vision_info(messages)
                data_scale = processor(text=[text], images=images, videos=videos, return_tensors='pt')
                data_scale = data_scale.to(device)

                try:
                    output_ids = model.generate(
                        **data_scale,
                        do_sample=False,
                        temperature=None,
                        top_p=None,
                        top_k=None,
                        repetition_penalty=None,
                        max_new_tokens=256)
                except RuntimeError as e:
                    if 'out of memory' in str(e).lower():
                        torch.cuda.empty_cache()
                    raise

                output_ids = output_ids[0, data_scale.input_ids.size(1):]
                if output_ids[-1] == processor.tokenizer.eos_token_id:
                    output_ids = output_ids[:-1]
                response = processor.decode(output_ids, clean_up_tokenization_spaces=False)
                print(f'   Response: {response}')

                all_responses.append(response)
                all_scale_names.append(scale['name'])

                if len(model.reg) > 0 and '<|reg|>' in response:
                    # Extract the most recent regression output.
                    blob = model.reg[-1].cpu().float()
                    pred_round = blob[:, :2] * duration
                    conf_round = blob[:, -1].tolist()
                    pred_round = pred_round.clamp(min=0, max=duration)

                    all_predictions.append(pred_round.clone())
                    all_confidences.append(conf_round.copy())

                    print(f'   Top-1: [{pred_round[0, 0].item():.1f}s, {pred_round[0, 1].item():.1f}s], '
                          f'conf={conf_round[0]:.3f}')
                    print(f'   Total candidates: {len(conf_round)}')

                    # Dual-signal early stopping: if consecutive predictions converge in both
                    # temporal overlap (IoU) and confidence score, terminate refinement early.
                    if len(all_predictions) >= 2:
                        prev_pred = all_predictions[-2][0]
                        curr_pred = all_predictions[-1][0]

                        intersection = max(0, min(curr_pred[1], prev_pred[1]) - max(curr_pred[0], prev_pred[0]))
                        union = max(curr_pred[1], prev_pred[1]) - min(curr_pred[0], prev_pred[0])
                        iou = intersection / union if union > 0 else 0

                        conf_diff = abs(all_confidences[-1][0] - all_confidences[-2][0])

                        print(f'   IoU with previous: {iou:.3f}, Conf diff: {conf_diff:.4f}')

                        if iou > 0.95 and conf_diff < 0.01:
                            print(f'   Early stopping: predictions converged at round {round_idx}')
                            break
                else:
                    print(f'   WARNING: Failed to generate timestamps at scale {scale["name"]}')
                    # If the very first scale fails, skip remaining scales.
                    if round_idx == 0:
                        break

                # Free GPU memory promptly after each scale round.
                del data_scale, images, videos
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # ==================== Cross-Scale Candidate Fusion ====================
            if len(all_confidences) > 0:
                print(f'\nMerging candidates from {len(all_confidences)} scale(s)...')

                K = 10  # Collect top-K candidates from each scale
                merged_pred = []
                merged_conf = []
                merged_scale_names = []

                for idx, (pred_list, conf_list, scale_name) in enumerate(
                        zip(all_predictions, all_confidences, all_scale_names)):
                    top_k = min(K, len(conf_list))
                    for i in range(top_k):
                        merged_pred.append(pred_list[i].tolist())
                        merged_conf.append(conf_list[i])
                        merged_scale_names.append(f'{scale_name}_rank{i+1}')

                print(f'   Total candidates before deduplication: {len(merged_pred)}')

                # IoU-based deduplication: segments with IoU > 0.9 are treated as duplicates.
                # When a duplicate is found, the higher-confidence candidate is retained.
                unique_indices = []
                for i in range(len(merged_pred)):
                    is_duplicate = False
                    for j in unique_indices:
                        pred_i = merged_pred[i]
                        pred_j = merged_pred[j]
                        intersection = max(0, min(pred_i[1], pred_j[1]) - max(pred_i[0], pred_j[0]))
                        union = max(pred_i[1], pred_j[1]) - min(pred_i[0], pred_j[0])
                        iou = intersection / union if union > 0 else 0

                        if iou > 0.9:
                            if merged_conf[i] > merged_conf[j]:
                                unique_indices.remove(j)
                                unique_indices.append(i)
                            is_duplicate = True
                            break

                    if not is_duplicate:
                        unique_indices.append(i)

                pred = [merged_pred[i] for i in unique_indices]
                conf = [merged_conf[i] for i in unique_indices]
                scale_names_filtered = [merged_scale_names[i] for i in unique_indices]

                print(f'   Total candidates after deduplication: {len(pred)}')
                print(f'   Top-5 candidates (before Verifier):')
                for i in range(min(5, len(pred))):
                    print(f'     {i+1}. [{pred[i][0]:.1f}s, {pred[i][1]:.1f}s] '
                          f'conf={conf[i]:.3f} from {scale_names_filtered[i]}')

                dump['grounder_response'] = all_responses[0]
                dump['grounder_success'] = True
                dump['refine_method'] = 'multi_scale_merged'
                dump['refine_all_scales'] = all_scale_names
                dump['refine_merged_candidates'] = len(pred)

                # Quantize predictions to dataset-specific time unit.
                unit = getattr(dataset_cls, 'UNIT', 0.001)
                pred_tensor = torch.tensor(pred)
                pred_tensor = torch.round(pred_tensor / unit).long() * unit
                pred = pred_tensor.tolist()
            else:
                # All scales failed; fall back to the full video span.
                print('WARNING: All scales failed to generate timestamps. Falling back to full video span.')
                pred = [[0, duration]]
                conf = [0]
                scale_names_filtered = ['fallback']

            print(pred[0], span, duration)
            dump['pred'] = pred
            dump['conf'] = conf

        # ==================== Verifier (Global Cross-Scale Reranking) ====================
        if do_grounding and adapter_state['verifier'] and len(pred) > 1:
            print('=============== verifier (Global Reranking) ===============')

            dump['agents'].append('verifier')

            probs = []
            for cand in pred[:20]:
                s0, e0 = parse_span(cand, duration, 2)
                offset = (e0 - s0) / 2
                s1, e1 = parse_span([s0 - offset, e0 + offset], duration)

                s = (s0 - s1) / (e1 - s1)
                e = (e0 - s1) / (e1 - s1)

                messages = [{
                    'role': 'user',
                    'content': [{
                        'type': 'video',
                        'video': video_path,
                        'num_threads': args.num_threads,
                        'video_start': s1,
                        'video_end': e1,
                        'min_pixels': 36 * 28 * 28,
                        'max_pixels': 64 * 28 * 28,
                        'max_frames': 64,
                        'fps': 2.0
                    }, {
                        'type': 'text',
                        'text': VERIFIER_PROMPT.format(question)
                    }]
                }]

                text = processor.apply_chat_template(messages, add_generation_prompt=True)
                images, videos = process_vision_info(messages)
                data = processor(text=[text], images=images, videos=videos, return_tensors='pt')

                video_grid_thw = data['video_grid_thw'][0]
                num_frames, window = int(video_grid_thw[0]), int(video_grid_thw[1] * video_grid_thw[2] / 4)

                pos_s, pos_e = round(s * num_frames), round(e * num_frames)
                pos_s, pos_e = min(max(0, pos_s), num_frames), min(max(0, pos_e), num_frames)

                base_idx = torch.nonzero(data['input_ids'][0] == model.config.vision_start_token_id).item()
                pos_s, pos_e = pos_s * window + base_idx + 1, pos_e * window + base_idx + 2

                input_ids = data['input_ids'][0].tolist()
                input_ids.insert(pos_s, model.config.seg_s_token_id)
                input_ids.insert(pos_e, model.config.seg_e_token_id)
                data['input_ids'] = torch.LongTensor([input_ids])
                data['attention_mask'] = torch.ones_like(data['input_ids'])

                data = data.to(device)

                model.base_model.disable_adapter_layers()
                model.base_model.enable_adapter_layers()
                model.set_adapter('verifier')

                try:
                    with torch.inference_mode():
                        logits = model(**data).logits[0, -1].softmax(dim=-1)
                except RuntimeError as e:
                    if 'out of memory' in str(e).lower():
                        torch.cuda.empty_cache()
                    raise

                score = (logits[9454] - logits[2753]).sigmoid().item()
                probs.append(score)

            probs_var = max(probs) - min(probs)
            print(f'Verifier score variance: {probs_var:.3f}')

            # Rerank all merged candidates by Verifier score (descending).
            ranks = torch.Tensor(probs).argsort(descending=True).tolist()
            pred = [pred[idx] for idx in ranks]
            conf_grounder = [conf[idx] for idx in ranks]
            conf_verifier = [probs[idx] for idx in ranks]
            scale_sources = [scale_names_filtered[idx] for idx in ranks]

            print(f'\nTop-5 after Verifier reranking:')
            for i in range(min(5, len(pred))):
                print(f'   {i+1}. [{pred[i][0]:.1f}s, {pred[i][1]:.1f}s] '
                      f'verifier={conf_verifier[i]:.3f} grounder={conf_grounder[i]:.3f} '
                      f'from {scale_sources[i]}')

            # Use Verifier scores as the final confidence signal.
            conf = conf_verifier

            dump['probs'] = probs
            dump['conf_grounder'] = conf_grounder
            dump['conf_verifier'] = conf_verifier
            dump['scale_sources'] = scale_sources
            dump['pred'] = pred
            dump['conf'] = conf

        # ==================== Answerer ====================
        if do_answering:
            print('=============== answerer ===============')

            dump['agents'].append('answerer')

            selected = pred[0] if 'pred' in dump else [0, duration]

            if hasattr(dataset_cls, 'MIN_RATIO'):
                min_len = duration * dataset_cls.MIN_RATIO
            else:
                min_len = getattr(dataset_cls, 'MIN_LEN', 32)

            s, e = parse_span(selected, duration, min_len)
            print([s, e], span, duration)

            if args.use_subtitle and 'subtitle_path' in anno and nncore.is_file(anno['subtitle_path']):
                subs = load_subtitle(anno['subtitle_path'])
                subs = [f'{round(a - s, 1)}s - {round(b - s, 1)}s, {t}\n' for a, b, t in subs if a >= s and b <= e]
                subs = ''.join(subs[:100])
                prompt = f'You are given a video with {round(e - s, 1)} seconds long.\nSubtitles:\n{subs}' + prompt

            messages = [{
                'role': 'user',
                'content': [{
                    'type': 'video',
                    'video': video_path,
                    'num_threads': args.num_threads,
                    'video_start': s,
                    'video_end': e,
                    'min_pixels': getattr(dataset_cls, 'MIN_PIXELS', 128) * 28 * 28,
                    'max_pixels': getattr(dataset_cls, 'MAX_PIXELS', 256) * 28 * 28,
                    'max_frames': getattr(dataset_cls, 'MAX_FRAMES', 32),
                    'fps': getattr(dataset_cls, 'FPS', 2.0)
                }, {
                    'type': 'text',
                    'text': prompt
                }]
            }]

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(device)

            text = processor.apply_chat_template(messages, add_generation_prompt=True)
            text += 'Best Option: (' if args.style == 'mcq' else ''
            print(text)
            images, videos = process_vision_info(messages)
            data = processor(text=[text], images=images, videos=videos, return_tensors='pt')
            data = data.to(device)

            if adapter_state['answerer']:
                model.base_model.disable_adapter_layers()
                model.base_model.enable_adapter_layers()
                model.set_adapter('answerer')
                context = nullcontext
            else:
                context = model.disable_adapter

            with context():
                try:
                    output_ids = model.generate(
                        **data,
                        do_sample=False,
                        temperature=None,
                        top_p=None,
                        top_k=None,
                        repetition_penalty=None,
                        max_new_tokens=256)
                except RuntimeError as e:
                    if 'out of memory' in str(e).lower():
                        torch.cuda.empty_cache()
                    raise

            assert data.input_ids.size(0) == output_ids.size(0) == 1
            output_ids = output_ids[0, data.input_ids.size(1):]
            if output_ids[-1] == processor.tokenizer.eos_token_id:
                output_ids = output_ids[:-1]
            response = processor.decode(output_ids, clean_up_tokenization_spaces=False)
            print(response)

            dump['answerer_response'] = response
            dump['response'] = response

        dumps.append(dump)

    nncore.dump(dumps, pred_path)