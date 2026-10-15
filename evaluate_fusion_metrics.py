import os
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from sewar.full_ref import vifp
from scipy.signal import convolve2d
from typing import Tuple, List, Dict, Optional

# 类型别名定义，增强代码可读性
Image = np.ndarray
MetricScores = Tuple[float, float, float, float]
TaskResult = Dict[str, float | str | int]


class FusionMetricsEvaluator:
    """图像融合指标评估器，封装所有指标计算逻辑"""

    # Record the metric definitions alongside every evaluation output.
    metric_version = "2026-09-20-fusion-metrics-v3"
    definition_source = "https://github.com/Zhaozixiang1228/MMIF-CDDFuse/blob/main/utils/Evaluator.py"

    @staticmethod
    def read_gray(path: str) -> Image:
        """读取灰度图像并保持0-255像素范围"""
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"无法读取图像: {path}")
        return img.astype(np.float32)

    @staticmethod
    def cal_EI(img: Image) -> float:
        """计算边缘强度(Edge Intensity)"""
        values = np.asarray(img, dtype=np.float64)
        gx = cv2.Sobel(values, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(values, cv2.CV_64F, 0, 1, ksize=3)
        return float(np.mean(np.hypot(gx, gy)))

    @staticmethod
    def cal_VIF(fused_img: Image, source1_img: Image, source2_img: Image) -> float:
        """计算视觉信息保真度(Visual Information Fidelity)"""
        # sewar.vifp(reference, distorted); fusion convention sums the two
        # source-reference scores. Do not average or reverse their roles.
        vif1 = vifp(source1_img, fused_img)
        vif2 = vifp(source2_img, fused_img)
        return float(vif1 + vif2)

    @staticmethod
    def cal_SCD(fused_img: Image, source1_img: Image, source2_img: Image) -> float:
        """计算相关差异和(Sum of Correlation Difference)"""
        fused = np.asarray(fused_img, dtype=np.float64)
        source1 = np.asarray(source1_img, dtype=np.float64)
        source2 = np.asarray(source2_img, dtype=np.float64)
        corr1 = np.corrcoef(source1.ravel(), (fused - source2).ravel())[0, 1]
        corr2 = np.corrcoef(source2.ravel(), (fused - source1).ravel())[0, 1]
        return float(corr1 + corr2)

    @staticmethod
    def cal_AG(img: Image) -> float:
        """计算平均梯度(Average Gradient)"""
        gy, gx = np.gradient(np.asarray(img, dtype=np.float64))
        return float(np.mean(np.sqrt((gx * gx + gy * gy) / 2.0)))

    @staticmethod
    def cal_MI(fused_img: Image, source1_img: Image, source2_img: Image) -> float:
        """Fusion MI in nats; VIFB-style per-image range normalization/256 bins."""
        def bins(image):
            values = np.asarray(image, dtype=np.float64)
            extent = np.ptp(values)
            if extent == 0:
                return np.zeros(values.shape, dtype=np.int64)
            scaled = (values - values.min()) * (255.0 / extent)
            # MATLAB positive integer casts round ties away from zero.
            return np.floor(scaled + 0.5).astype(np.int64).clip(0, 255)

        def mutual_information(a, b):
            counts = np.bincount((a.ravel() * 256 + b.ravel()), minlength=65536).reshape(256, 256)
            joint = counts / counts.sum()
            independent = joint.sum(axis=1)[:, None] * joint.sum(axis=0)[None, :]
            nonzero = joint > 0
            return np.sum(joint[nonzero] * np.log(joint[nonzero] / independent[nonzero]))

        fused = bins(fused_img)
        return float(mutual_information(bins(source1_img), fused) +
                     mutual_information(bins(source2_img), fused))

    @staticmethod
    def cal_QABF(fused_img: Image, source1_img: Image, source2_img: Image) -> float:
        """Edge-transfer score with min/max strength ratio and zero-padded Sobel.

        Equal nonzero magnitudes have ratio 1; do not reproduce the magnitude-
        instead-of-ratio equality branch found in several public code copies.
        """
        hx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
        hy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float64)

        def edges(image):
            values = np.asarray(image, dtype=np.float64)
            gx = convolve2d(values, hx, mode='same', boundary='fill')
            gy = convolve2d(values, hy, mode='same', boundary='fill')
            angle = np.full(values.shape, np.pi / 2)
            nonzero = gx != 0
            angle[nonzero] = np.arctan(gy[nonzero] / gx[nonzero])
            return np.hypot(gx, gy), angle

        gf, af = edges(fused_img)
        numerator, denominator = 0.0, 0.0
        for source in [source1_img, source2_img]:
            gs, a_s = edges(source)
            maximum = np.maximum(gs, gf)
            strength = np.divide(np.minimum(gs, gf), maximum,
                                 out=np.ones_like(maximum), where=maximum > 0)
            orientation = 1.0 - np.abs(a_s - af) / (np.pi / 2)
            quality_g = 0.9994 / (1.0 + np.exp(-15.0 * (strength - 0.5)))
            quality_a = 0.9879 / (1.0 + np.exp(-22.0 * (orientation - 0.8)))
            numerator += np.sum(gs * quality_g * quality_a)
            denominator += np.sum(gs)
        return float(numerator / denominator) if denominator else float('nan')

    def extended_metrics(self, source1: Image, source2: Image, fused: Image) -> Dict[str, float]:
        if source1.shape != source2.shape or source1.shape != fused.shape:
            raise ValueError('All arrays must have the same shape')
        return {
            'EI': self.cal_EI(fused), 'VIF': self.cal_VIF(fused, source1, source2),
            'SCD': self.cal_SCD(fused, source1, source2), 'AG': self.cal_AG(fused),
            'MI_nats': self.cal_MI(fused, source1, source2),
            'QABF': self.cal_QABF(fused, source1, source2),
        }

    def calc_metrics(self, source1_path: str, source2_path: str, fused_path: str) -> MetricScores:
        """计算单组图像的四个评估指标"""
        source1 = self.read_gray(source1_path)
        source2 = self.read_gray(source2_path)
        fused = self.read_gray(fused_path)

        if source1.shape != source2.shape or fused.shape != source1.shape:
            raise ValueError("Source and fused image dimensions must match")

        return (
            self.cal_EI(fused),
            self.cal_VIF(fused, source1, source2),
            self.cal_SCD(fused, source1, source2),
            self.cal_AG(fused)
        )

    def evaluate_task(
            self,
            task_name: str,
            source1_dir: str,
            source2_dir: str,
            fused_dir: str,
            output_dir: Optional[str] = None
    ) -> Optional[TaskResult]:
        """评估单个任务的所有图像并返回结果"""
        print(f"\n🔹 评估任务: {task_name}")

        if not os.path.exists(fused_dir):
            print(f"⚠️ 跳过任务：融合目录不存在 - {fused_dir}")
            return None

        # 获取并筛选有效的图像文件
        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
        fused_files = sorted([
            f for f in os.listdir(fused_dir)
            if f.lower().endswith(valid_extensions)
        ])

        metrics: Dict[str, List[float]] = {
            "EI": [], "VIF": [], "SCD": [], "AG": []
        }
        per_image = []
        failures = []

        for filename in fused_files:
            source1_path = os.path.join(source1_dir, filename)
            source2_path = os.path.join(source2_dir, filename)
            fused_path = os.path.join(fused_dir, filename)

            # 检查源图像是否存在
            if not all(os.path.exists(p) for p in [source1_path, source2_path]):
                print(f"⚠️ 跳过 {filename}：源图像不完整")
                failures.append({"image": filename, "reason": "missing source image"})
                continue

            try:
                ei, vif, scd, ag = self.calc_metrics(source1_path, source2_path, fused_path)
                if not np.all(np.isfinite([ei, vif, scd, ag])):
                    raise ValueError("Non-finite metric: inspect constant images or numerical degeneracy")
                metrics["EI"].append(ei)
                metrics["VIF"].append(vif)
                metrics["SCD"].append(scd)
                metrics["AG"].append(ag)
                per_image.append({"image": filename, "EI": ei, "VIF": vif, "SCD": scd, "AG": ag})
                print(f"✅ 完成 {filename}：EI={ei:.2f}, VIF={vif:.2f}, SCD={scd:.2f}, AG={ag:.2f}")
            except Exception as e:
                print(f"❌ 处理 {filename} 失败：{str(e)}")
                failures.append({"image": filename, "reason": str(e)})
                continue

        # 计算平均值
        sample_count = len(metrics["EI"])
        if sample_count == 0:
            print(f"⚠️ 任务 {task_name} 无有效样本")
            return None

        avg_metrics = {
            "Task": task_name,
            "Sample_Count": sample_count,
            "EI": np.mean(metrics["EI"]),
            "VIF": np.mean(metrics["VIF"]),
            "SCD": np.mean(metrics["SCD"]),
            "AG": np.mean(metrics["AG"])
        }
        self.last_per_image = pd.DataFrame(per_image)
        self.last_per_image.attrs["metric_version"] = self.metric_version
        if output_dir is not None:
            destination = Path(output_dir)
            destination.mkdir(parents=True, exist_ok=True)
            self.last_per_image.to_csv(destination / "per_image.csv", index=False, encoding="utf-8")
            summary = {
                **avg_metrics, "metric_version": self.metric_version,
                "definition_source": self.definition_source,
                "vif_implementation": "sewar.full_ref.vifp; source reference first; sum of two scores",
                "ei_implementation": "mean 3x3 Sobel gradient magnitude; OpenCV default reflected border",
                "ag_implementation": "mean sqrt((dx^2+dy^2)/2); centered interior and one-sided boundary differences",
                "candidate_images": len(fused_files), "excluded_images": failures,
            }
            (destination / "summary.json").write_text(
                json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        # 输出任务平均结果
        print(f"\n📊 {task_name} 平均指标：")
        print(f"EI={avg_metrics['EI']:.4f}, VIF={avg_metrics['VIF']:.4f}, "
              f"SCD={avg_metrics['SCD']:.4f}, AG={avg_metrics['AG']:.4f}")

        return avg_metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate existing fusion outputs")
    parser.add_argument('--source1', required=True)
    parser.add_argument('--source2', required=True)
    parser.add_argument('--fused', required=True)
    parser.add_argument('--task', default='fusion')
    parser.add_argument('--output-dir', required=True)
    options = parser.parse_args()
    result = FusionMetricsEvaluator().evaluate_task(
        options.task, options.source1, options.source2, options.fused, options.output_dir)
    if result is None:
        raise SystemExit('No images were evaluated')


if __name__ == "__main__":
    main()
