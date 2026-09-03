"""星点检测注入-回收实验命令。"""

from __future__ import annotations

import argparse
from pathlib import Path

from .experiments import (
    run_crowded_blend_audit,
    run_feature_audit,
    run_pair_flux_ratio_audit,
    run_injection_recovery,
    run_proposal_mode_comparison,
    run_real_background_injection,
    run_stratified_real_background_injection,
    write_injection_artifacts,
    write_crowded_blend_audit_artifacts,
    write_feature_audit_artifacts,
    write_pair_flux_ratio_audit_artifacts,
    write_proposal_mode_comparison_artifacts,
    write_real_background_injection_artifacts,
    write_stratified_real_background_injection_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="运行 rst19 星点检测的可复现注入-回收实验")
    parser.add_argument("--out-dir", type=Path, default=Path("tmp") / "injection", help="实验输出目录")
    parser.add_argument(
        "--feature-audit",
        action="store_true",
        help="运行孤立/弱源/双源/尖峰/掩膜/边缘/饱和/长线分层控制实验",
    )
    parser.add_argument("--trials", type=int, default=4, help="每个强度档位的随机试验数")
    parser.add_argument("--sources", type=int, default=24, help="每次试验注入源数")
    parser.add_argument(
        "--peak-levels",
        nargs="+",
        type=float,
        help="显式指定注入峰值 ADU 档位；未指定时使用各模式默认值",
    )
    parser.add_argument("--threshold-sigma", type=float, default=4.0, help="匹配滤波候选阈值 sigma")
    parser.add_argument("--min-flux-snr", type=float, default=5.0, help="质量层孔径通量 SNR 下限")
    parser.add_argument("--psf-fwhm", type=float, default=3.0, help="注入和检测使用的 Gaussian PSF FWHM（pixel）")
    parser.add_argument("--injected-psf-fwhm", type=float, help="合成背景模式的注入 PSF FWHM；默认与检测核相同")
    parser.add_argument("--pair-separation-px", type=float, help="公平比较时按双星注入的分离距离；--sources 必须为偶数")
    parser.add_argument(
        "--pair-flux-ratio-audit",
        action="store_true",
        help="在真实 FITS 上固定双源总峰值并扫描 secondary/primary 强弱比；必须配合 --real-fits",
    )
    parser.add_argument(
        "--total-peak-levels",
        nargs="+",
        type=float,
        help="双源强弱比审计的总控制信号 ADU 档位；peak_excess 时为总峰值超额，integrated_excess 时为总离散积分超额",
    )
    parser.add_argument(
        "--secondary-ratios",
        nargs="+",
        type=float,
        default=(1.0, 0.5, 0.25, 0.125),
        help="双源强弱比 secondary/primary；默认 1 0.5 0.25 0.125",
    )
    parser.add_argument(
        "--pairs",
        type=int,
        default=4,
        help="双源强弱比审计每次试验的双源对数；与 --sources 不同",
    )
    parser.add_argument("--proposal-mode", choices=("gaussian", "hybrid", "ensemble"), default="hybrid", help="Gaussian、Gaussian+DoG，或再加入 starlet 小波")
    parser.add_argument("--min-distance", type=int, default=4, help="候选峰最小间距（pixel）")
    parser.add_argument("--min-psf-support-pixels", type=int, default=3, help="候选峰中心 3×3 内的最少 PSF 支持像素数")
    parser.add_argument(
        "--real-fits",
        type=Path,
        help="可选真实 FITS；提供后在原图真实背景上运行实验，而不是使用均匀合成背景",
    )
    parser.add_argument(
        "--compare-proposals",
        action="store_true",
        help="在同一真实背景、同一注入坐标上公平比较 Gaussian 与 hybrid；必须配合 --real-fits",
    )
    parser.add_argument(
        "--psf-model",
        choices=("gaussian", "empirical"),
        default="gaussian",
        help="真实 FITS 模式的注入 PSF；empirical 从首帧质量源提取实测 PSF",
    )
    parser.add_argument(
        "--pair-signal-normalization",
        choices=("peak_excess", "integrated_excess"),
        default="peak_excess",
        help="双源强弱比审计固定峰值超额，或固定离散注入小窗积分超额",
    )
    parser.add_argument(
        "--crowded-blend-audit",
        action="store_true",
        help="在真实 FITS 上审计 2/3 源拥挤组的合并、解析和多分裂；必须配合 --real-fits",
    )
    parser.add_argument(
        "--blend-sizes",
        nargs="+",
        type=int,
        default=(2, 3),
        help="拥挤组内真值源数；默认 2 3",
    )
    parser.add_argument(
        "--blend-separations",
        nargs="+",
        type=float,
        default=(3.0, 5.4, 6.0),
        help="拥挤组相邻真值间距 pixel；默认 3 5.4 6",
    )
    parser.add_argument(
        "--blend-total-levels",
        nargs="+",
        type=float,
        default=(256.0, 512.0),
        help="拥挤组总控制信号 ADU；默认 256 512",
    )
    parser.add_argument(
        "--blend-groups",
        type=int,
        default=4,
        help="每个拥挤条件每次试验的注入组数；默认 4",
    )
    parser.add_argument(
        "--blend-signal-normalization",
        choices=("peak_excess", "integrated_excess"),
        default="integrated_excess",
        help="拥挤组固定峰值超额，或固定离散注入小窗积分超额",
    )
    parser.add_argument(
        "--stratified-real",
        action="store_true",
        help="在真实 FITS 的背景/边缘/拥挤/特殊值域条件上运行分层注入审计",
    )
    parser.add_argument(
        "--strata",
        nargs="+",
        choices=("blank", "high_background", "edge", "crowded", "special_code", "line"),
        default=("blank", "high_background", "edge", "crowded", "special_code"),
        help="分层真实背景注入的条件层；line 需要基线检测中存在线状候选",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.feature_audit:
        rows = run_feature_audit(
            threshold_sigma=args.threshold_sigma,
            min_distance=args.min_distance,
            min_flux_snr=args.min_flux_snr,
            psf_fwhm=args.psf_fwhm,
            min_psf_support_pixels=args.min_psf_support_pixels,
            proposal_mode=args.proposal_mode,
        )
        output = write_feature_audit_artifacts(rows, args.out_dir)
        print(f"feature audit: {output}")
        for row in rows:
            print(
                f"{row.scenario}: control={row.control_type} "
                f"candidate_hits={row.candidate_true_hits} quality_hits={row.quality_true_hits} "
                f"nearby={row.nearby_candidate_count}/{row.nearby_quality_count}"
            )
        return 0
    if args.compare_proposals and args.real_fits is None:
        raise SystemExit("--compare-proposals 必须配合 --real-fits")
    if args.stratified_real and args.real_fits is None:
        raise SystemExit("--stratified-real 必须配合 --real-fits")
    if args.stratified_real and args.compare_proposals:
        raise SystemExit("--stratified-real 不能与 --compare-proposals 同时使用")
    if args.pair_flux_ratio_audit and args.real_fits is None:
        raise SystemExit("--pair-flux-ratio-audit 必须配合 --real-fits")
    if args.crowded_blend_audit and args.real_fits is None:
        raise SystemExit("--crowded-blend-audit 必须配合 --real-fits")
    if sum(bool(value) for value in (args.pair_flux_ratio_audit, args.crowded_blend_audit, args.compare_proposals, args.stratified_real)) > 1:
        raise SystemExit("--pair-flux-ratio-audit/--crowded-blend-audit/--compare-proposals/--stratified-real 只能选择一个")
    if args.peak_levels is not None and any(level <= 0 for level in args.peak_levels):
        raise SystemExit("--peak-levels 必须全部为正数")
    if args.total_peak_levels is not None and any(level <= 0 for level in args.total_peak_levels):
        raise SystemExit("--total-peak-levels 必须全部为正数")
    if any(level <= 0 for level in args.blend_total_levels):
        raise SystemExit("--blend-total-levels 必须全部为正数")
    if any(size < 2 for size in args.blend_sizes):
        raise SystemExit("--blend-sizes 必须全部至少为 2")
    if any(separation <= 0 for separation in args.blend_separations):
        raise SystemExit("--blend-separations 必须全部为正数")
    if any(ratio <= 0 or ratio > 1 for ratio in args.secondary_ratios):
        raise SystemExit("--secondary-ratios 必须位于 (0, 1]")
    if args.pairs < 1:
        raise SystemExit("--pairs 必须为正数")
    if args.blend_groups < 1:
        raise SystemExit("--blend-groups 必须为正数")
    if args.crowded_blend_audit:
        progress_state = {"last": (-1, -1)}

        def report_blend_progress(index: int, total: int) -> None:
            state = (int(index), int(total))
            if state != progress_state["last"]:
                progress_state["last"] = state
                print(f"crowded-blend progress: {index}/{total}", flush=True)

        rows = run_crowded_blend_audit(
            args.real_fits,
            group_sizes=tuple(args.blend_sizes),
            separations_px=tuple(args.blend_separations),
            total_control_levels=tuple(args.blend_total_levels),
            trials_per_condition=args.trials,
            groups_per_trial=args.blend_groups,
            threshold_sigma=args.threshold_sigma,
            min_flux_snr=args.min_flux_snr,
            psf_fwhm=args.psf_fwhm,
            injected_psf_fwhm=args.injected_psf_fwhm,
            proposal_mode=args.proposal_mode,
            min_distance=args.min_distance,
            min_psf_support_pixels=args.min_psf_support_pixels,
            psf_model=args.psf_model,
            signal_normalization=args.blend_signal_normalization,
            progress=report_blend_progress,
        )
        output = write_crowded_blend_audit_artifacts(rows, args.out_dir)
        print(f"crowded blend audit: {output}")
        for row in rows:
            print(
                f"size={row.group_size} separation={row.nearest_separation_px:g} "
                f"total={row.total_control_signal_adu:g} ({row.signal_normalization}) "
                f"source={row.source_candidate_recall:.3f}/{row.source_quality_recall:.3f} "
                f"group={row.group_candidate_resolution_recall:.3f}/{row.group_quality_resolution_recall:.3f} "
                f"merged={row.merged_candidate_fraction:.3f}/{row.merged_quality_fraction:.3f} "
                f"extra={row.extra_candidate_fraction:.3f}/{row.extra_quality_fraction:.3f}"
            )
        return 0
    if args.pair_flux_ratio_audit:
        progress_state = {"last": (-1, -1)}

        def report_pair_progress(index: int, total: int) -> None:
            state = (int(index), int(total))
            if state != progress_state["last"]:
                progress_state["last"] = state
                print(f"pair-ratio progress: {index}/{total}", flush=True)

        rows = run_pair_flux_ratio_audit(
            args.real_fits,
            total_peak_levels=(
                tuple(args.total_peak_levels)
                if args.total_peak_levels is not None
                else (128.0, 256.0, 512.0)
            ),
            secondary_to_primary_ratios=tuple(args.secondary_ratios),
            trials_per_condition=args.trials,
            pairs_per_trial=args.pairs,
            pair_separation_px=(args.pair_separation_px if args.pair_separation_px is not None else 5.4),
            threshold_sigma=args.threshold_sigma,
            min_flux_snr=args.min_flux_snr,
            psf_fwhm=args.psf_fwhm,
            injected_psf_fwhm=args.injected_psf_fwhm,
            proposal_mode=args.proposal_mode,
            min_distance=args.min_distance,
            min_psf_support_pixels=args.min_psf_support_pixels,
            psf_model=args.psf_model,
            signal_normalization=args.pair_signal_normalization,
            progress=report_pair_progress,
        )
        output = write_pair_flux_ratio_audit_artifacts(rows, args.out_dir)
        print(f"pair flux-ratio audit: {output}")
        for row in rows:
            print(
                f"total={row.total_control_signal_adu:g} ({row.signal_normalization}) "
                f"ratio={row.secondary_to_primary_ratio:g} "
                f"source={row.source_candidate_recall:.3f}/{row.source_quality_recall:.3f} "
                f"pair={row.pair_candidate_resolution_recall:.3f}/{row.pair_quality_resolution_recall:.3f} "
                f"merged={row.merged_candidate_fraction:.3f}/{row.merged_quality_fraction:.3f}"
            )
        return 0
    if args.compare_proposals:
        rows = run_proposal_mode_comparison(
            args.real_fits,
            peak_levels=(
                tuple(args.peak_levels)
                if args.peak_levels is not None
                else (12.0, 24.0, 56.0)
            ),
            trials_per_level=args.trials,
            sources_per_trial=args.sources,
            threshold_sigma=args.threshold_sigma,
            min_flux_snr=args.min_flux_snr,
            psf_fwhm=args.psf_fwhm,
            injected_psf_fwhm=args.injected_psf_fwhm,
            pair_separation_px=args.pair_separation_px,
            min_distance=args.min_distance,
            min_psf_support_pixels=args.min_psf_support_pixels,
        )
        output = write_proposal_mode_comparison_artifacts(rows, args.out_dir)
        print(f"same-position proposal comparison: {output}")
        for row in rows:
            print(
                f"peak={row.peak_excess_adu:g} "
                f"candidate(g={row.gaussian_candidate_recall:.3f}, h={row.hybrid_candidate_recall:.3f}, e={row.ensemble_candidate_recall:.3f}) "
                f"quality(g={row.gaussian_quality_recall:.3f}, h={row.hybrid_quality_recall:.3f}, e={row.ensemble_quality_recall:.3f})"
            )
        return 0
    if args.stratified_real:
        progress_state = {"last": (-1, -1)}

        def report_progress(index: int, total: int) -> None:
            state = (int(index), int(total))
            if state != progress_state["last"]:
                progress_state["last"] = state
                print(f"stratified progress: {index}/{total}", flush=True)

        rows = run_stratified_real_background_injection(
            args.real_fits,
            strata=tuple(args.strata),
            peak_levels=(
                tuple(args.peak_levels)
                if args.peak_levels is not None
                else (24.0, 56.0, 128.0)
            ),
            trials_per_level=args.trials,
            sources_per_trial=args.sources,
            threshold_sigma=args.threshold_sigma,
            min_flux_snr=args.min_flux_snr,
            psf_fwhm=args.psf_fwhm,
            proposal_mode=args.proposal_mode,
            min_distance=args.min_distance,
            min_psf_support_pixels=args.min_psf_support_pixels,
            psf_model=args.psf_model,
            progress=report_progress,
        )
        output = write_stratified_real_background_injection_artifacts(rows, args.out_dir)
        print(f"stratified real-background injection recovery: {output}")
        for row in rows:
            print(
                f"stratum={row.stratum} peak={row.peak_excess_adu:g} "
                f"candidate={row.candidate_recall:.3f} quality={row.quality_recall:.3f} "
                f"strict={row.unambiguous_candidate_recall if row.unambiguous_candidate_recall is not None else 'n/a'}/"
                f"{row.unambiguous_quality_recall if row.unambiguous_quality_recall is not None else 'n/a'} "
                f"unambiguous={row.unambiguous_injected_count}/{row.injected_count} "
                f"ambiguous={row.ambiguous_injection_count}/{row.injected_count}"
            )
        return 0
    if args.real_fits is not None:
        rows = run_real_background_injection(
            args.real_fits,
            peak_levels=(
                tuple(args.peak_levels)
                if args.peak_levels is not None
                else (12.0, 16.0, 24.0, 36.0, 56.0, 84.0, 128.0)
            ),
            trials_per_level=args.trials,
            sources_per_trial=args.sources,
            threshold_sigma=args.threshold_sigma,
            min_flux_snr=args.min_flux_snr,
            psf_fwhm=args.psf_fwhm,
            proposal_mode=args.proposal_mode,
            min_distance=args.min_distance,
            min_psf_support_pixels=args.min_psf_support_pixels,
            psf_model=args.psf_model,
        )
        output = write_real_background_injection_artifacts(rows, args.out_dir)
        print(f"real-background injection recovery: {output}")
    else:
        rows = run_injection_recovery(
            peak_levels=(
                tuple(args.peak_levels)
                if args.peak_levels is not None
                else (8.0, 12.0, 16.0, 24.0, 36.0, 56.0, 84.0, 128.0)
            ),
            trials_per_level=args.trials,
            sources_per_trial=args.sources,
            threshold_sigma=args.threshold_sigma,
            min_flux_snr=args.min_flux_snr,
            psf_fwhm=args.psf_fwhm,
            injected_psf_fwhm=args.injected_psf_fwhm,
            proposal_mode=args.proposal_mode,
            min_distance=args.min_distance,
            min_psf_support_pixels=args.min_psf_support_pixels,
        )
        output = write_injection_artifacts(rows, args.out_dir)
        print(f"injection recovery: {output}")
    for row in rows:
        print(f"peak={row.peak_excess_adu:g} candidate={row.candidate_recall:.3f} quality={row.quality_recall:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
