import tomllib
import numpy as np
from dataclasses import dataclass
from typing import Union, List, Tuple
import cv2  # 统一cv2导入方式，避免分散调用
from pathlib import Path  # 增强路径处理能力


@dataclass(frozen=True)
class Config:
    # 必要配置
    type: str
    bpm: Union[int, float]
    img_path: Path
    time: int
    # 输出配置
    output_path: Path
    allow_0ms: bool
    # 模式配置（根据type动态生效）
    xy_range: Tuple[Tuple[float, float], Tuple[float, float]]
    xz_range: Tuple[Tuple[float, float], float]
    xz_y_pos: float
    yz_range: Tuple[Tuple[float, float], float]
    yz_x_pos: float
    freemode_is_beats: bool
    freemode_quad: Tuple[Tuple[float, float, float], ...]

@dataclass
class Point:
    x: float
    y: float
    z: Union[float, int, None] = None

    def __post_init__(self):
        for coord in [self.x, self.y, self.z]:
            if coord is not None and not np.isfinite(coord):
                raise ValueError(f"无效坐标值: x={self.x}, y={self.y}, z={self.z}")

@dataclass
class ArcTrace:
    startTime: int  # 起始时间（ms）
    startX: float   # 起始X坐标
    startY: float   # 起始Y坐标
    endTime: int    # 结束时间（ms）
    endX: float     # 结束X坐标
    endY: float     # 结束Y坐标


def load_config(config_path: Union[str, Path] = "config.toml") -> Config:
    """
    加载并解析配置文件，返回结构化的Config对象
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        结构化配置对象
        
    Raises:
        FileNotFoundError: 配置文件不存在
        KeyError: 配置项缺失
        ValueError: 配置值无效
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path.absolute()}")
    
    # 加载原始配置
    with open(config_path, "rb") as f:
        raw_config = tomllib.load(f)
    
    # 验证必要配置项
    necessary_keys = ["type", "bpm", "path", "time"]
    if not all(key in raw_config.get("necessary", {}) for key in necessary_keys):
        raise KeyError(f"必要配置项缺失，需包含: {necessary_keys}")
    
    optional = raw_config.get("optional", {})
    mode_config = raw_config.get(raw_config["necessary"]["type"], {})
    

    try:
        return Config(
            # 必要配置
            type=raw_config["necessary"]["type"],
            bpm=raw_config["necessary"]["bpm"],
            img_path=Path(raw_config["necessary"]["path"]),
            time=raw_config["necessary"]["time"],
            # 输出配置
            output_path=Path(optional.get("output_path", "arc_output.aff")),
            allow_0ms=optional.get("allow_0ms", False),
            # 模式配置
            xy_range=mode_config.get("xy_range", ((0.0, 1.0), (0.0, 1.0))),
            xz_range=mode_config.get("xz_range", ((0.0, 1.0), 1.0)),
            xz_y_pos=mode_config.get("y_position", 0.0),
            yz_range=mode_config.get("yz_range", ((0.0, 1.0), 1.0)),
            yz_x_pos=mode_config.get("x_position", 0.0),
            freemode_is_beats=mode_config.get("is_beats_mode", False),
            freemode_quad=tuple(mode_config.get("p"+str(i), ()) for i in range(4))
        )
    except KeyError as e:
        raise KeyError(f"配置项缺失: {e}") from e
    except TypeError as e:
        raise ValueError(f"配置值类型错误: {e}") from e

def map_to_quadrilateral(u: float, v: float, quad: Tuple[Tuple[float, ...], ...]) -> Tuple[float, float, float]:
    """
    将单位矩形内的点(u, v)映射到3D四边形（双线性插值）
    
    Args:
        u: 单位矩形X坐标（0≤u≤1）
        v: 单位矩形Y坐标（0≤v≤1）
        quad: 四边形4个顶点，顺序为(p0, p1, p2, p3)，每个顶点为(x,y,z)
        
    Returns:
        映射后的3D坐标(x, y, z)
    """
    # 确保u/v在有效范围（避免图像边缘误差导致的越界）
    u_clamped = max(0.0, min(1.0, u))
    v_clamped = max(0.0, min(1.0, v))
    
    # 双线性插值（向量化计算，比循环更高效）
    p0, p1, p2, p3 = [np.array(pt, dtype=np.float64) for pt in quad]
    bottom_edge = (1 - u_clamped) * p0 + u_clamped * p1
    top_edge = (1 - u_clamped) * p3 + u_clamped * p2
    mapped = (1 - v_clamped) * bottom_edge + v_clamped * top_edge
    
    return tuple(mapped.tolist())

class ImageToArcTraceConverter:
    """图像转ArcTrace转换器，职责单一，支持多种模式"""
    
    def __init__(self, config: Config):
        """
        初始化转换器
        
        Args:
            config: 结构化配置对象
        """
        self.config = config
        self.ms_addition = 0 if config.allow_0ms else 1  # 提前计算，避免重复计算
        self._validate_config()  # 初始化时验证配置
    
    def _validate_config(self):
        """验证配置有效性（提前暴露错误，避免运行中崩溃）"""
        # 验证模式合法性
        valid_modes = ["XY", "XZ", "YZ", "FREEMODE"]
        if self.config.type not in valid_modes:
            raise ValueError(f"无效模式: {self.config.type}，支持模式: {valid_modes}")
        
        # 验证范围配置（最小值<最大值）
        mode = self.config.type
        if mode == "XY":
            x_min, x_max = self.config.xy_range[0]
            y_min, y_max = self.config.xy_range[1]
            if x_min >= x_max or y_min >= y_max:
                raise ValueError(f"XY范围无效: x({x_min},{x_max}), y({y_min},{y_max})（最小值必须小于最大值）")
        elif mode == "XZ":
            x_min, x_max = self.config.xz_range[0]
            if x_min >= x_max or self.config.xz_range[1] <= 0:
                raise ValueError(f"XZ范围无效: x({x_min},{x_max}), z_scale({self.config.xz_range[1]})（x最小值<最大值，z_scale>0）")
        elif mode == "YZ":
            y_min, y_max = self.config.yz_range[0]
            if y_min >= y_max or self.config.yz_range[1] <= 0:
                raise ValueError(f"YZ范围无效: y({y_min},{y_max}), z_scale({self.config.yz_range[1]})（y最小值<最大值，z_scale>0）")
        elif mode == "FREEMODE":
            if len(self.config.freemode_quad) != 4:
                raise ValueError(f"FREEMODE四边形顶点数量无效: {len(self.config.freemode_quad)}（需4个顶点）")
            for i, pt in enumerate(self.config.freemode_quad):
                if len(pt) != 3:
                    raise ValueError(f"FREEMODE顶点{i}无效: {pt}（需3D坐标(x,y,z)）")
        
        # 验证BPM有效性
        if self.config.bpm <= 0:
            raise ValueError(f"BPM无效: {self.config.bpm}（必须大于0）")
        
        # 验证图像路径有效性
        if not self.config.img_path.exists():
            raise FileNotFoundError(f"图像文件不存在: {self.config.img_path.absolute()}")
    
    def _preprocess_image(self, img_path: Path, epsilon: float) -> List[np.ndarray]:
        """
        图像预处理：读取→灰度→腐蚀→膨胀→二值化→提取轮廓
        
        Args:
            img_path: 图像路径
            epsilon: 轮廓近似精度（用于approxPolyDP）
        
        Returns:
            提取到的轮廓列表（每个轮廓为np.ndarray）
        """
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"无法读取图像: {img_path.absolute()}（可能是格式不支持或文件损坏）")
        
        # 图像预处理流水线（使用局部变量减少属性访问开销）
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((4, 4), np.uint8)
        eroded = cv2.erode(gray, kernel, iterations=1)
        dilated = cv2.dilate(eroded, kernel, iterations=1)
        
        # 二值化（THRESH_OTSU自动计算阈值，适应不同亮度图像）
        _, binary = cv2.threshold(dilated, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 提取轮廓（RETR_TREE保留层级，CHAIN_APPROX_SIMPLE压缩轮廓点）
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # 轮廓近似（减少轮廓点数量，优化后续计算）
        return [cv2.approxPolyDP(contour, epsilon, True) for contour in contours]
    
    def _convert_contours_to_points(self, contours: List[np.ndarray], img_shape: Tuple[int, int]) -> List[List[Point]]:
        """
        将轮廓转换为单位矩形内的Point列表（归一化到[0,1]）
        
        Args:
            contours: 预处理后的轮廓列表
            img_shape: 图像形状（高度，宽度）
        
        Returns:
            轮廓点列表（每个轮廓对应一个Point列表）
        """
        img_height, img_width = img_shape
        contour_points = []
        
        for contour in contours:
            # 轮廓点归一化：x→[0,1]（图像宽度方向），y→[0,1]（图像高度方向，翻转y轴）
            points = [
                Point(
                    x=float(pt[0][0]) / img_width,
                    y=1.0 - float(pt[0][1]) / img_height  # 图像坐标系y轴向下，翻转后向上
                )
                for pt in contour
            ]
            # 过滤重复点（避免相邻点相同导致无效轨迹）
            filtered_points = []
            for p in points:
                if not filtered_points or not (np.isclose(p.x, filtered_points[-1].x) and np.isclose(p.y, filtered_points[-1].y)):
                    filtered_points.append(p)
            if len(filtered_points) >= 2:  # 至少2个点才构成轨迹
                contour_points.append(filtered_points)
        
        return contour_points
    
    def _generate_arc_traces(self, contour_points: List[List[Point]]) -> List[ArcTrace]:
        """
        根据模式生成ArcTrace列表（核心逻辑模块化，按模式拆分）
        
        Args:
            contour_points: 轮廓点列表
        
        Returns:
            ArcTrace列表
        """
        arc_traces = []
        mode = self.config.type
        bpm = self.config.bpm
        ms_per_beat = 60000.0 / float(bpm)  # 提前计算每拍毫秒数，避免重复计算
        
        for points in contour_points:
            n = len(points)
            for i in range(n):
                p1 = points[i]
                p2 = points[(i + 1) % n]  # 封闭轮廓：最后一个点连回第一个点
                
                # 按模式生成轨迹
                if mode == "XY":
                    arc_traces.append(self._generate_xy_trace(p1, p2))
                elif mode == "XZ":
                    trace = self._generate_xz_trace(p1, p2, ms_per_beat)
                    if trace:
                        arc_traces.append(trace)
                elif mode == "YZ":
                    trace = self._generate_yz_trace(p1, p2, ms_per_beat)
                    if trace:
                        arc_traces.append(trace)
                elif mode == "FREEMODE":
                    trace = self._generate_freemode_trace(p1, p2, ms_per_beat)
                    if trace:
                        arc_traces.append(trace)
        
        return arc_traces
    

    def _generate_xy_trace(self, p1: Point, p2: Point) -> ArcTrace:
        """生成XY模式的ArcTrace"""
        x_min, x_max = self.config.xy_range[0]
        y_min, y_max = self.config.xy_range[1]
        x_scale = x_max - x_min
        y_scale = y_max - y_min
        
        return ArcTrace(
            startTime=self.config.time,
            startX=x_min + p1.x * x_scale,
            startY=y_min + p1.y * y_scale,
            endTime=self.config.time + self.ms_addition,
            endX=x_min + p2.x * x_scale,
            endY=y_min + p2.y * y_scale
        )
    
    def _generate_xz_trace(self, p1: Point, p2: Point, ms_per_beat: float) -> Union[ArcTrace, None]:
        """生成XZ模式的ArcTrace，返回None如果点无效"""
        # 跳过相同点
        if np.isclose(p1.x, p2.x) and np.isclose(p1.y, p2.y):
            return None
        
        # 确保p1.y ≤ p2.y（按z轴顺序排序）
        if p1.y > p2.y:
            p1, p2 = p2, p1
        
        x_min, x_max = self.config.xz_range[0]
        x_scale = x_max - x_min
        z_scale = self.config.xz_range[1]
        y_pos = self.config.xz_y_pos
        
        # 计算时间（z轴对应时间）
        start_time = self.config.time + int(p1.y * ms_per_beat * z_scale)
        end_time = self.config.time + int(p2.y * ms_per_beat * z_scale)
        # 相同z轴时添加时间偏移（避免0ms时长）
        end_time += self.ms_addition if np.isclose(p1.y, p2.y) else 0
        
        return ArcTrace(
            startTime=start_time,
            startX=x_min + p1.x * x_scale,
            startY=y_pos,
            endTime=end_time,
            endX=x_min + p2.x * x_scale,
            endY=y_pos
        )
    
    def _generate_yz_trace(self, p1: Point, p2: Point, ms_per_beat: float) -> Union[ArcTrace, None]:
        """生成YZ模式的ArcTrace，返回None如果点无效"""
        # 跳过相同点
        if np.isclose(p1.x, p2.x) and np.isclose(p1.y, p2.y):
            return None
        
        # 确保p1.x ≤ p2.x（按z轴顺序排序）
        if p1.x > p2.x:
            p1, p2 = p2, p1
        
        y_min, y_max = self.config.yz_range[0]
        y_scale = y_max - y_min
        z_scale = self.config.yz_range[1]
        x_pos = self.config.yz_x_pos
        
        # 计算时间（z轴对应时间）
        start_time = self.config.time + int(p1.x * ms_per_beat * z_scale)
        end_time = self.config.time + int(p2.x * ms_per_beat * z_scale)
        # 相同z轴时添加时间偏移（避免0ms时长）
        end_time += self.ms_addition if np.isclose(p1.x, p2.x) else 0
        
        return ArcTrace(
            startTime=start_time,
            startX=x_pos,
            startY=y_min + p1.y * y_scale,
            endTime=end_time,
            endX=x_pos,
            endY=y_min + p2.y * y_scale
        )
    
    def _generate_freemode_trace(self, p1: Point, p2: Point, ms_per_beat: float) -> Union[ArcTrace, None]:
        """生成FREEMODE模式的ArcTrace，返回None如果点无效"""
        # 跳过相同点
        if np.isclose(p1.x, p2.x) and np.isclose(p1.y, p2.y):
            return None
        
        # 映射到3D四边形
        mapped1 = map_to_quadrilateral(p1.x, p1.y, self.config.freemode_quad)
        mapped2 = map_to_quadrilateral(p2.x, p2.y, self.config.freemode_quad)
        m1 = Point(*mapped1)
        m2 = Point(*mapped2)
        
        # 确保m1.z ≤ m2.z（按z轴顺序排序）
        if m1.z > m2.z:
            m1, m2 = m2, m1
        
        # 计算时间（根据是否为节拍模式）
        if self.config.freemode_is_beats:
            start_time = self.config.time + int(m1.z * ms_per_beat)
            end_time = self.config.time + int(m2.z * ms_per_beat)
        else:
            start_time = int(m1.z)
            end_time = int(m2.z)
        
        # 相同z轴时添加时间偏移（避免0ms时长）
        end_time += self.ms_addition if np.isclose(m1.z, m2.z) else 0
        
        return ArcTrace(
            startTime=start_time,
            startX=m1.x,
            startY=m1.y,
            endTime=end_time,
            endX=m2.x,
            endY=m2.y
        )
    

    def process(self, epsilon: float = None) -> List[ArcTrace]:
        """
        完整处理流程：图像预处理→轮廓转点→生成轨迹
        
        Args:
            epsilon: 轮廓近似精度（默认根据模式自动设置）
        
        Returns:
            生成的ArcTrace列表
        """
        if epsilon is None:
            epsilon_map = {"XY": 3.0, "XZ": 5.0, "YZ": 5.0, "FREEMODE": 1.0}
            epsilon = epsilon_map.get(self.config.type, 1.0)
        
        # 1. 图像预处理
        img = cv2.imread(str(self.config.img_path))
        img_shape = img.shape[:2]  # (高度, 宽度)
        contours = self._preprocess_image(self.config.img_path, epsilon)
        
        # 2. 轮廓转单位矩形点
        contour_points = self._convert_contours_to_points(contours, img_shape)
        if not contour_points:
            raise Warning("未提取到有效轮廓，返回空轨迹列表")
        
        # 3. 生成ArcTrace
        return self._generate_arc_traces(contour_points)
    
    def save_aff(self, arc_traces: List[ArcTrace], out_path: Union[str, Path] = None):
        """
        保存ArcTrace到.aff文件
        
        Args:
            arc_traces: 要保存的ArcTrace列表
            out_path: 输出路径（默认使用配置中的output_path）
        """
        out_path = Path(out_path) if out_path else self.config.output_path
        
        # 确保输出目录存在
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 写入文件（使用f-string格式化，比%更清晰）
        with open(out_path, "w", encoding="utf-8") as f:
            for trace in arc_traces:
                f.write(
                    f"arc({trace.startTime},{trace.endTime},{trace.startX:.2f},{trace.endX:.2f},s,"
                    f"{trace.startY:.2f},{trace.endY:.2f},0,none,true);\n"
                )


def main(config_path: Union[str, Path] = "config.toml"):
    """
    主函数：加载配置→初始化转换器→处理→保存
    
    Args:
        config_path: 配置文件路径
    """
    try:
        # 加载配置
        config = load_config(config_path)
        print(f"成功加载配置，模式: {config.type}，图像: {config.img_path.name}")
        
        # 初始化转换器并处理
        converter = ImageToArcTraceConverter(config)
        arc_traces = converter.process()
        
        # 保存结果
        converter.save_aff(arc_traces)
        print(f"处理完成！生成{len(arc_traces)}条轨迹，保存到: {config.output_path.absolute()}")
    
    except (FileNotFoundError, KeyError, ValueError, Warning) as e:
        print(f"处理失败: {e}")
        exit(1)


if __name__ == "__main__":
    # 支持命令行传入配置文件路径（示例：python script.py custom_config.toml）
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.toml"
    main(config_path)