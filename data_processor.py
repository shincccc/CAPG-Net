# data_processor.py (V67 每日Nino直读版)
import numpy as np
import pandas as pd


class DataProcessor:
    """
    数据处理器 V67
    功能：
    1. 加载 EOP (UT1)
    2. 分别加载 AAM, OAM, HAM, SLAM 并合成 Total EAM
    3. 执行精细化物理分离: LOD_core = LOD_obs - (AAM+OAM+HAM+SLAM)
    4. 对齐 ENSO 数据
    """

    def __init__(self, config):
        self.config = config

    def load_all_data(self):

        print("📥 加载并预处理多源物理数据 (DataProcessor V67)...")

        # 1. 调用新的混合加载函数
        eop_df = self._load_hybrid_eop_data()

        # 2. 加载并合并激发函数 (AAM+OAM+HAM+SLAM)
        excitation_df = self._load_and_merge_excitations()

        # 3. 加载 ENSO 数据
        nino_df = self._load_nino_data(eop_df['date'].min(), eop_df['date'].max())

        # 4. 加载地磁数据 (Kp/Ap/F10.7)
        geo_df = self._load_geomagnetic_data()

        # 5. 对齐四方数据时间范围
        eop_df, nino_df, excitation_df, geo_df = self._align_data_ranges(
            eop_df, nino_df, excitation_df, geo_df
        )

        # 6. 执行物理分离
        print("⚡ 执行精细化物理分离: LOD_core = LOD_obs - (AAM+OAM+HAM+SLAM) ...")
        eop_df = self._compute_physics_derived_features(eop_df, excitation_df)

        print(f"✅ 数据预处理完成. 样本数: {len(eop_df)}")

        # 返回四个对齐后的 DataFrame
        return eop_df, nino_df, excitation_df, geo_df

    def _load_geomagnetic_data(self):
        """
        [新增] 加载地磁数据 (Kp, Ap, F10.7)
        """
        try:
            # 尝试从 Config 读取路径，如果没有则使用默认路径
            path = getattr(self.config, 'GEOMAGNETIC_FILE', r'D:\eop\UT1_Prediction\data\Geomagnetic\Kp_Ap_daily.csv')

            geo = pd.read_csv(path)
            geo['date'] = pd.to_datetime(geo['date'])
            geo.set_index('date', inplace=True)

            # 简单的缺失值处理
            geo = geo.fillna(method='ffill').fillna(method='bfill')

            # 计算 Ap 的 30 天滑动平均 (反映地磁长期活跃度)
            geo['ap_ma30'] = geo['ap'].rolling(window=30, min_periods=1).mean()

            return geo
        except Exception as e:
            print(f"⚠️ 警告: 加载地磁数据失败 ({e})，将使用全 0 填充。")
            # 返回一个空的 DataFrame 结构，防止后续报错
            return pd.DataFrame(columns=['kp', 'ap', 'f107', 'sn', 'ap_ma30'])

    def _align_data_ranges(self, eop_df, nino_df, ex_df, geo_df):
        """
        对齐四个数据集的时间索引 (EOP, Nino, Excitation, Geomagnetic)
        """
        # 1. 确定公共时间窗口
        # 注意：如果有某个数据缺失(如geo_df为空)，需要做容错
        start_dates = [eop_df['date'].min(), nino_df.index.min(), ex_df.index.min()]
        end_dates = [eop_df['date'].max(), nino_df.index.max(), ex_df.index.max()]

        if not geo_df.empty:
            start_dates.append(geo_df.index.min())
            end_dates.append(geo_df.index.max())

        common_start = max(d for d in start_dates if pd.notna(d))
        common_end = min(d for d in end_dates if pd.notna(d))

        # 预留 Buffer (用于差分计算)
        calc_start = common_start - pd.Timedelta(days=7)

        # 2. 切片 EOP
        eop_df = eop_df[(eop_df['date'] >= calc_start) & (eop_df['date'] <= common_end)].copy()

        # 3. 初步对齐其他数据
        nino_df = nino_df.loc[nino_df.index.isin(eop_df['date'])]
        ex_df = ex_df.loc[ex_df.index.isin(eop_df['date'])]
        if not geo_df.empty:
            geo_df = geo_df.loc[geo_df.index.isin(eop_df['date'])]

        # 4. 反向过滤 EOP (取交集)
        valid_dates = ex_df.index.intersection(nino_df.index)
        if not geo_df.empty:
            valid_dates = valid_dates.intersection(geo_df.index)

        eop_df = eop_df[eop_df['date'].isin(valid_dates)].sort_values('date').reset_index(drop=True)

        # 5. 最终严格重置索引 (填充缺失值为0，防止对齐失败)
        nino_df = nino_df.reindex(eop_df['date']).fillna(0.0)
        ex_df = ex_df.reindex(eop_df['date']).fillna(0.0)

        if not geo_df.empty:
            geo_df = geo_df.reindex(eop_df['date']).fillna(0.0)
        else:
            # 如果加载失败，生成全0数据
            geo_df = pd.DataFrame(0.0, index=eop_df['date'], columns=['kp', 'ap', 'f107', 'sn', 'ap_ma30'])

        return eop_df, nino_df, ex_df, geo_df

    def _load_eop_data(self):
        """加载 EOP (UT1) 数据"""
        try:
            eop = pd.read_csv(self.config.EOP_FILE)
        except FileNotFoundError:
            raise FileNotFoundError(f"找不到EOP文件: {self.config.EOP_FILE}")

        # 标准化列名
        eop.columns = [c.lower() for c in eop.columns]
        if 'ut1r-tai' in eop.columns:
            eop.rename(columns={'ut1r-tai': 'ut1'}, inplace=True)
        elif 'ut1-tai' in eop.columns:
            eop.rename(columns={'ut1-tai': 'ut1'}, inplace=True)

        # 解析日期
        if 'date' in eop.columns:
            eop['date'] = pd.to_datetime(eop['date'])
        else:
            # 兼容 Year, Month, Day 格式
            eop['date'] = pd.to_datetime(eop[['year', 'month', 'day']])

        eop_df = eop[['date', 'ut1']].sort_values('date').reset_index(drop=True)
        return eop_df

    def _load_hybrid_eop_data(self):
        """
        [核心逻辑] 融合 C04 和 Finals
        策略：以 C04 为基准，用 Finals 填充 C04 结束之后的空白。
        """
        # A. 加载 C04 (Definitive)
        if not hasattr(self.config, 'EOP_FILE'):
            raise ValueError("Config 缺少 EOP_FILE 路径")
        df_c04 = self._read_standard_csv(self.config.EOP_FILE)

        # B. 加载 Finals (Rapid)
        if hasattr(self.config, 'FINALS_FILE'):
            df_finals = self._read_standard_csv(self.config.FINALS_FILE)
        else:
            print("⚠️ 未配置 FINALS_FILE，仅使用 C04 数据")
            return df_c04

        # C. 执行严谨融合
        # 1. 找到 C04 的最后一天
        last_c04_date = df_c04['date'].max()

        # 2. 截取 Finals 中晚于 C04 的部分 (Strictly newer)
        # 注意：这里我们隐含地信任 Finals 在重叠期之后的连续性
        new_data = df_finals[df_finals['date'] > last_c04_date].copy()

        if len(new_data) > 0:
            print(
                f"   🔗 检测到 Finals 新数据: {len(new_data)} 天 (从 {new_data['date'].min().date()} 到 {new_data['date'].max().date()})")

            # 3. 拼接
            combined_df = pd.concat([df_c04, new_data], axis=0).sort_values('date').reset_index(drop=True)
            return combined_df
        else:
            print("   ℹ️ Finals 没有比 C04 更新的数据，仅使用 C04。")
            return df_c04

    def _load_and_merge_excitations(self):
        """加载 4 个独立激发函数文件并合并"""
        # 定义分量名称与配置路径的映射
        components = {
            'aam': self.config.AAM_FILE,
            'oam': self.config.OAM_FILE,
            'ham': self.config.HAM_FILE,
            'slam': self.config.SLAM_FILE
        }

        merged_df = None

        for name, path in components.items():
            # 目标列名 (如 AAM_ms)
            target_col = f"{name.upper()}_ms"

            # 加载单文件
            df = self._load_single_excitation(path, target_col, name)

            if merged_df is None:
                merged_df = df
            else:
                # 外连接合并，保留最大时间跨度
                merged_df = merged_df.join(df, how='outer')

        # 缺失值处理：线性插值填补中间空缺，0值填补两端
        merged_df = merged_df.interpolate(method='time').fillna(0.0)

        # 计算总激发 (Total EAM)
        # 注意：这里假设各分量已经统一为 ms 单位 (由 make_xxx_daily.py 保证)
        merged_df['total_eam'] = (merged_df['aam'] + merged_df['oam'] +
                                  merged_df['ham'] + merged_df['slam'])

        return merged_df

    def _load_single_excitation(self, filepath, target_col, rename_to):
        """
        通用加载器
        filepath: 文件路径
        target_col: CSV中包含数据的列名 (如 'AAM_ms')
        rename_to: DataFrame中重命名后的列名 (如 'aam')
        """
        try:
            df = pd.read_csv(filepath)

            # 自动解析日期 (兼容 Date 列或 MJD)
            if 'Date' in df.columns:
                df['date'] = pd.to_datetime(df['Date'])
            elif 'MJD' in df.columns:
                origin = pd.Timestamp('1858-11-17')
                df['date'] = origin + pd.to_timedelta(df['MJD'], unit='D')
            else:
                # 尝试 Year/Month/Day
                df['date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])

            # 提取数据列
            if target_col not in df.columns:
                # 尝试模糊匹配 (兼容旧格式或列名不一致)
                candidates = [c for c in df.columns if rename_to.upper() in c and 'ms' in c]
                if candidates:
                    target_col = candidates[0]
                else:
                    # 最后的兜底：找最后一列
                    target_col = df.columns[-1]
                    print(f"⚠️ 警告: {filepath} 未找到列 {target_col}，使用最后一列 {target_col} 代替。")

            # 设置索引并重采样 (去噪)
            df = df.set_index('date')[[target_col]].rename(columns={target_col: rename_to})

            # 强制按日重采样并简单去噪 (3天均值，消除高频抖动)
            df = df.resample('D').mean().rolling(window=3, center=True, min_periods=1).mean()

            return df

        except Exception as e:
            print(f"❌ 加载 {rename_to.upper()} 失败 ({filepath}): {e}")
            # 返回空 DataFrame 避免程序崩溃，但会全是 0
            return pd.DataFrame()

    def _load_nino_data(self, min_date, max_date):
        """
        加载 ENSO 数据 (V4：直接读取现成的日分辨率数据)
        替代了之前V3版本中复杂的因果外推法，彻底消除了数据泄露的风险。
        """
        try:
            # 1. 读取日数据文件
            nino = pd.read_csv(self.config.NINO34_FILE)
            nino['date'] = pd.to_datetime(nino['date'])
            nino = nino.set_index('date').sort_index()

            # 2. 提取目标列并确保是数值类型
            # 请根据你的CSV文件列名修改此处，常见的可能是 'nino34' 或 'NINO34'
            target_col = 'nino34'
            if target_col not in nino.columns:
                # 尝试查找包含'nino'的列名
                col_candidates = [c for c in nino.columns if 'nino' in c.lower()]
                if col_candidates:
                    target_col = col_candidates[0]
                    print(f"   ℹ️ 未找到列名 'nino34'，自动使用: '{target_col}'")
                else:
                    raise KeyError(f"在文件中找不到NINO3.4数据列。可用列: {nino.columns.tolist()}")

            nino_daily = nino[[target_col]].astype(float)

            print(f"   📅 ENSO日数据加载成功: {len(nino_daily)} 天")
            return nino_daily

        except FileNotFoundError:
            print("⚠️ 警告: 找不到ENSO日数据文件，使用全0填充")
            idx = pd.date_range(min_date, max_date)
            return pd.DataFrame({'nino34': 0.0}, index=idx)

    def _compute_physics_derived_features(self, eop_df, ex_df):
        """
        计算物理特征
        LOD_core = LOD_obs - Total_EAM
        """
        # 1. 计算观测日长 (LOD_obs)
        ut1_vals = eop_df['ut1'].values
        # LOD = -(UT1_t - UT1_{t-1})
        lod_obs = -1.0 * np.diff(ut1_vals, prepend=ut1_vals[0]) * 1000.0  # s -> ms
        lod_obs[0] = lod_obs[1]  # 修正首位差分
        eop_df['lod_obs'] = lod_obs

        # 2. 提取 Total EAM 序列(全量)
        # Lag = 1 天 (表示大气超前1天，昨天的风影响今天的转速)
        # Scale = 1.1162 (表示理论模型幅度偏小，需要放大1.1倍)
        OPTIMAL_LAG = 1
        ADMITTANCE_FACTOR = 1.1162
        total_eam_series = ex_df['total_eam'].shift(OPTIMAL_LAG).fillna(0.0)
        total_eam = total_eam_series.values * ADMITTANCE_FACTOR

        # 3. 物理减法 (LOD_core)
        # 理论上：LOD_obs = LOD_core + LOD_fluid + Noise
        # 所以：LOD_core = LOD_obs - LOD_fluid (即 Total EAM)
        eop_df['lod_core'] = lod_obs - total_eam_series.values

        # 4. 核心积分 (UT1_core)
        eop_df['ut1_core_integral'] = np.cumsum(eop_df['lod_core']) / 1000.0

        # 5. 计算长期趋势 (用于长周期预报)
        # 使用 365 天滑动平均提取地核的长期漂移
        eop_df['lod_core_trend_365'] = eop_df['lod_core'].rolling(window=365, min_periods=1).mean()

        # 最终清洗 NaN
        eop_df.fillna(0.0, inplace=True)

        return eop_df

    def _read_standard_csv(self, filepath):
        """辅助函数：读取并标准化 CSV (统一列名为 lower case)"""
        try:
            df = pd.read_csv(filepath)
            # 统一列名处理
            df.columns = [c.lower() for c in df.columns]

            # 映射 UT1 列名
            if 'ut1r-tai' in df.columns:
                df.rename(columns={'ut1r-tai': 'ut1'}, inplace=True)
            elif 'ut1-tai' in df.columns:
                df.rename(columns={'ut1-tai': 'ut1'}, inplace=True)

            # 解析日期
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            else:
                # 兼容 Year/Month/Day 分列的情况
                if {'year', 'month', 'day'}.issubset(df.columns):
                    df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
                elif 'mjd' in df.columns:
                    # 如果只有 MJD，也可以转换
                    origin = pd.Timestamp('1858-11-17')
                    df['date'] = origin + pd.to_timedelta(df['mjd'], unit='D')

            return df[['date', 'ut1']].dropna().sort_values('date')

        except Exception as e:
            print(f"❌ 读取文件失败 {filepath}: {e}")
            return pd.DataFrame(columns=['date', 'ut1'])