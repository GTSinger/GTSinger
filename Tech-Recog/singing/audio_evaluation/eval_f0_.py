import numpy as np
import tqdm
from tabulate import tabulate

def gross_pitch_error(true_t, true_f, est_t, est_f):
    """The relative frequency in percent of pitch estimates that are
    outside a threshold around the true pitch. Only frames that are
    considered pitched by both the ground truth and the estimator (if
    applicable) are considered.
    """

    correct_frames = _true_voiced_frames(true_t, true_f, est_t, est_f)
    gross_pitch_error_frames = _gross_pitch_error_frames(
        true_t, true_f, est_t, est_f
    )
    return np.sum(gross_pitch_error_frames) / np.sum(correct_frames)


def _gross_pitch_error_frames(true_t, true_f, est_t, est_f, eps=1e-8):
    voiced_frames = _true_voiced_frames(true_t, true_f, est_t, est_f)
    true_f_p_eps = [x + eps for x in true_f]
    pitch_error_frames = np.abs(est_f / true_f_p_eps - 1) > 0.2
    return voiced_frames & pitch_error_frames

def _mse_pitch_error_frames(true_t, true_f, est_t, est_f):
    voiced_frames = _true_voiced_frames(true_t, true_f, est_t, est_f)
    diff = np.square(true_f - est_f) * np.array(voiced_frames, dtype=np.float)
    return diff

def _true_voiced_frames(true_t, true_f, est_t, est_f):
    return (est_f != 0) & (true_f != 0)

def _voicing_decision_error_frames(true_t, true_f, est_t, est_f):
    return (est_f != 0) != (true_f != 0)

def f0_frame_error(true_t, true_f, est_t, est_f):
    gross_pitch_error_frames = _gross_pitch_error_frames(
        true_t, true_f, est_t, est_f
    )
    voicing_decision_error_frames = _voicing_decision_error_frames(
        true_t, true_f, est_t, est_f
    )
    return (np.sum(gross_pitch_error_frames) +
            np.sum(voicing_decision_error_frames)) / (len(true_f))


def voicing_decision_error(true_t, true_f, est_t, est_f):
    voicing_decision_error_frames = _voicing_decision_error_frames(
        true_t, true_f, est_t, est_f
    )
    return np.sum(voicing_decision_error_frames) / (len(true_f))

def f0_mse_error(true_t, true_f, est_t, est_f):
    correct_frames = _true_voiced_frames(true_t, true_f, est_t, est_f)
    f0_mse_error_frames = _mse_pitch_error_frames(true_t, true_f, est_t, est_f)
    return np.sqrt(np.sum(f0_mse_error_frames) / np.sum(correct_frames))

def eval_f0_error(samples, distortion_fn):
    results = []
    for sample in tqdm.tqdm(samples):
        if sample is None:
            results.append(None)
            continue
        # assume single channel
        yref_f = sample["gt"]
        ysyn_f = sample["cond"]

        yref_f = np.array(yref_f)
        ysyn_f = np.array(ysyn_f)

        distortion = distortion_fn(None, yref_f, None, ysyn_f)
        results.append((distortion.item(),
                        len(yref_f),
                        len(ysyn_f)
                        ))
    return results

def eval_gross_pitch_error(samples):
    return eval_f0_error(samples, gross_pitch_error)


def eval_voicing_decision_error(samples):
    return eval_f0_error(samples, voicing_decision_error)


def eval_f0_frame_error(samples):
    return eval_f0_error(samples, f0_frame_error)

def eval_f0_mse_error(samples):
    return eval_f0_error(samples, f0_mse_error)

def print_results(results, show_bin):
    results = np.array(list(filter(lambda x: x is not None, results)))

    np.set_printoptions(precision=3)

    def _print_result(results):
        res = {
            "nutt": len(results),
            "error": results[:, 0].mean(),
            "std": results[:, 0].std(),
            "dur_ref": int(results[:, 1].sum()),
            "dur_syn": int(results[:, 2].sum()),
        }
        print(tabulate([res.values()], res.keys(), floatfmt=".4f"))

    print(">>>> ALL")
    _print_result(results)

    if show_bin:
        edges = [0, 200, 400, 600, 800, 1000, 2000, 4000]
        for i in range(1, len(edges)):
            mask = np.logical_and(results[:, 1] >= edges[i-1],
                                  results[:, 1] < edges[i])
            if not mask.any():
                continue
            bin_results = results[mask]
            print(f">>>> ({edges[i-1]}, {edges[i]})")
            _print_result(bin_results)


def main(eval_f0, gpe, vde, ffe, f0rmse, show_bin):
    f0_samples = np.load(eval_f0, allow_pickle=True)

    if gpe:
        print("===== Evaluate Gross Pitch Error =====")
        results = eval_gross_pitch_error(f0_samples)
        print_results(results, show_bin)
    if vde:
        print("===== Evaluate Voicing Decision Error =====")
        results = eval_voicing_decision_error(f0_samples)
        print_results(results, show_bin)
    if ffe:
        print("===== Evaluate F0 Frame Error =====")
        results = eval_f0_frame_error(f0_samples)
        print_results(results, show_bin)
    if f0rmse:
        print("===== Evaluate F0 Frame Error =====")
        results = eval_f0_mse_error(f0_samples)
        print_results(results, show_bin)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("eval_f0")
    parser.add_argument("--gpe", action="store_true")
    parser.add_argument("--vde", action="store_true")
    parser.add_argument("--ffe", action="store_true")
    parser.add_argument("--f0rmse", action="store_true")
    parser.add_argument("--show-bin", action="store_true")
    args = parser.parse_args()

    main(args.eval_f0, args.gpe, args.vde, args.ffe, args.f0rmse, args.show_bin)