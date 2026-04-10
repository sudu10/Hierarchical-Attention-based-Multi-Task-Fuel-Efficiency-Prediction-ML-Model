import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.preprocessing import RobustScaler
import re
import warnings
warnings.filterwarnings('ignore')


class VEDFeatureEngineer:
    def __init__(self, window_size: int = 60, overlap: int = 30):
        self.window_size = window_size
        self.overlap = overlap
        self.scaler = RobustScaler()
        self.telemetry_scalers = {}
        self.stats = {
            'total_windows': 0,
            'skipped_short_trips': 0,
            'skipped_errors': 0,
            'per_vehicle': {}
        }

    def load_ved_data(self, signals_path: str, static_path: str, vehicle_id) -> Tuple[pd.DataFrame, Dict]:
        signals = pd.read_csv(signals_path)
        static = pd.read_excel(static_path)

        vehicle_signals = signals[signals['VehId'] == vehicle_id].copy()

        if len(vehicle_signals) == 0:
            raise ValueError(f"No signal data found for vehicle {vehicle_id}")

        static_row = static[static['VehId'] == vehicle_id]
        if len(static_row) == 0:
            raise ValueError(f"Vehicle {vehicle_id} not found in static data")

        raw_static = static_row.iloc[0].to_dict()
        vehicle_static = self._parse_static(raw_static, vehicle_signals)

        return vehicle_signals, vehicle_static

    def _parse_static(self, raw: Dict, signals: pd.DataFrame) -> Dict:
        disp_str = str(raw.get('Engine Configuration & Displacement', '2.0'))
        disp_match = re.search(r'(\d+\.?\d*)\s*[Ll]', disp_str)
        displacement = float(disp_match.group(1)) if disp_match else 2.0

        fuel_type_map = {
            'ICE': 'Gasoline', 'ICEV': 'Gasoline',
            'HEV': 'Hybrid', 'PHEV': 'PHEV', 'BEV': 'EV',
        }
        vtype = str(raw.get('Vehicle Type', 'ICEV')).strip()
        fuel_type = fuel_type_map.get(vtype, 'Gasoline')

        raw_class = str(raw.get('Vehicle Class', 'Car')).strip()
        class_map = {
            'Car': 'Car', 'Sedan': 'Car', 'Hatchback': 'Car',
            'Coupe': 'Car', 'Wagon': 'Car', 'Compact': 'Car',
            'SUV': 'SUV', 'Crossover': 'SUV', 'CUV': 'SUV',
            'Truck': 'Truck', 'Pickup': 'Truck', 'Van': 'Truck',
            'Minivan': 'Truck',
        }
        vehicle_class = class_map.get(raw_class, 'Car')

        highway = signals[
            (signals['Vehicle Speed[km/h]'] > 70) &
            (signals['Fuel Rate[L/hr]'] > 0.5)
        ]
        if len(highway) > 30:
            avg_speed = highway['Vehicle Speed[km/h]'].mean()
            avg_fuel = highway['Fuel Rate[L/hr]'].mean()
            if avg_fuel > 0:
                kpl = avg_speed / avg_fuel
                baseline_mpg = float(np.clip(kpl * 2.35215, 15, 60))
            else:
                baseline_mpg = 28.0
        else:
            baseline_mpg = 28.0

        weight = float(raw.get('Generalized_Weight', 1500))

        return {
            'Engine Displacement[L]': displacement,
            'Curb Weight[kg]': weight,
            'Fuel Type': fuel_type,
            'EPA Combined MPG': baseline_mpg,
            'Class': vehicle_class,
        }

    def calculate_acceleration(self, df: pd.DataFrame) -> np.ndarray:
        df = df.sort_values('DayNum').reset_index(drop=True)

        speed_ms = df['Vehicle Speed[km/h]'].fillna(0).values / 3.6

        time_days = df['DayNum'].fillna(method='ffill').fillna(0).values
        time_seconds = time_days * 86400.0

        for i in range(1, len(time_seconds)):
            if time_seconds[i] <= time_seconds[i - 1]:
                time_seconds[i] = time_seconds[i - 1] + 1.0

        dt = np.diff(time_seconds)
        dt = np.where(dt < 0.1, 1.0, dt)
        dv = np.diff(speed_ms)

        acceleration = np.zeros(len(speed_ms))
        acceleration[1:] = dv / dt

        acceleration = np.nan_to_num(acceleration, nan=0.0, posinf=0.0, neginf=0.0)
        acceleration = np.clip(acceleration, -10, 10)

        return acceleration

    def create_time_series_features(self, window: pd.DataFrame) -> np.ndarray:
        features = np.stack([
            window['Vehicle Speed[km/h]'].fillna(0).values,
            window['Engine RPM[RPM]'].fillna(0).values,
            window['Absolute Load[%]'].fillna(0).values,
            window['MAF[g/sec]'].fillna(0).values,
            window['Acceleration[m/s²]'].values,
            window['Fuel Rate[L/hr]'].fillna(0).values,
        ], axis=0)

        if features.shape[1] < self.window_size:
            pad_width = ((0, 0), (0, self.window_size - features.shape[1]))
            features = np.pad(features, pad_width, mode='edge')

        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        return features[:, :self.window_size].astype(np.float32)

    def create_vehicle_context(self, window: pd.DataFrame, static_info: Dict) -> np.ndarray:
        fuel_type_map = {'Gasoline': 0, 'Diesel': 1, 'Hybrid': 2, 'PHEV': 3, 'EV': 4}
        class_map = {'Car': 0, 'SUV': 1, 'Truck': 2}

        oat_col = 'OAT[DegC]' if 'OAT[DegC]' in window.columns else None
        ac_col = 'Air Conditioning Power[kW]' if 'Air Conditioning Power[kW]' in window.columns else None

        oat_mean = float(window[oat_col].mean()) if oat_col else 25.0
        ac_mean = float(window[ac_col].mean()) if ac_col else 0.0

        if np.isnan(oat_mean):
            oat_mean = 25.0
        if np.isnan(ac_mean):
            ac_mean = 0.0

        context = np.array([
            static_info.get('Engine Displacement[L]', 2.0),
            static_info.get('Curb Weight[kg]', 1500),
            fuel_type_map.get(static_info.get('Fuel Type', 'Gasoline'), 0),
            static_info.get('EPA Combined MPG', 28.0),
            class_map.get(static_info.get('Class', 'Car'), 0),
            oat_mean,
            ac_mean,
        ], dtype=np.float32)

        return np.nan_to_num(context, nan=0.0)

    def calculate_fuel_loss(self, window: pd.DataFrame, baseline_mpg: float) -> float:
        speed_kmh = window['Vehicle Speed[km/h]'].fillna(0).values
        fuel_rate_lhr = window['Fuel Rate[L/hr]'].fillna(0).values

        distance_km = (speed_kmh / 3600.0).sum()

        fuel_consumed_l = (fuel_rate_lhr / 3600.0).sum()

        if fuel_consumed_l > 1e-6 and distance_km > 1e-6:
            distance_miles = distance_km * 0.621371
            fuel_gallons = fuel_consumed_l * 0.264172
            observed_mpg = distance_miles / fuel_gallons
        else:
            observed_mpg = baseline_mpg

        fuel_loss = ((baseline_mpg - observed_mpg) / baseline_mpg) * 100.0
        return float(np.clip(fuel_loss, 0, 100))

    def classify_behavior(self, window: pd.DataFrame) -> int:
        speed = window['Vehicle Speed[km/h]'].fillna(0).values
        accel = window['Acceleration[m/s²]'].values
        rpm = window['Engine RPM[RPM]'].fillna(0).values

        harsh_accel = (accel > 1.5).sum()
        harsh_brake = (accel < -1.5).sum()
        strong_accel = (accel > 2.5).sum()
        strong_brake = (accel < -2.5).sum()
        high_rpm = (rpm > 2500).sum()
        very_high_rpm = (rpm > 3000).sum()
        idle_count = (speed < 5).sum()
        avg_rpm = rpm.mean()
        avg_speed = speed.mean()
        speed_std = speed.std() if len(speed) > 1 else 0

        speed_changes = (np.abs(np.diff(speed)) > 8).sum() if len(speed) > 1 else 0

        if very_high_rpm > 15 or avg_rpm > 2800 or high_rpm > 30:
            return 4

        if strong_brake > 3 or harsh_brake > 8:
            return 3

        if strong_accel > 3 or harsh_accel > 8:
            return 2

        if (speed_changes > 6 or
                (idle_count > 15 and speed_std > 12) or
                (idle_count > 20)):
            return 5

        if (harsh_accel < 3 and harsh_brake < 3 and
                avg_rpm < 2200 and idle_count < 5 and
                speed_std < 15 and avg_speed > 20):
            return 0

        return 1

    def calculate_route_efficiency(self, window: pd.DataFrame) -> float:
        speed = window['Vehicle Speed[km/h]'].fillna(0).values
        accel = window['Acceleration[m/s²]'].values

        n = len(speed)
        if n == 0:
            return 0.5

        avg_speed = speed.mean()
        speed_std = speed.std() if n > 1 else 0
        stop_count = (speed < 5).sum()
        smooth_ratio = (np.abs(accel) < 1.0).sum() / n

        speed_score = min(avg_speed / 80.0, 1.0)
        stability_score = max(1.0 - speed_std / 30.0, 0.0)
        flow_score = max(1.0 - stop_count / n, 0.0)
        smoothness_score = smooth_ratio

        efficiency = (0.3 * speed_score + 0.2 * stability_score +
                      0.3 * flow_score + 0.2 * smoothness_score)

        return float(np.clip(efficiency, 0, 1))

    def create_windows(self, signals: pd.DataFrame, static_info: Dict) -> List[Dict]:
        signals = signals.copy()

        signals['Acceleration[m/s²]'] = self.calculate_acceleration(signals)

        windows = []

        for trip_id in signals['Trip'].unique():
            trip_data = signals[signals['Trip'] == trip_id].reset_index(drop=True)

            if len(trip_data) < self.window_size:
                self.stats['skipped_short_trips'] += 1
                continue

            for start_idx in range(0, len(trip_data) - self.window_size + 1, self.overlap):
                end_idx = start_idx + self.window_size
                window = trip_data.iloc[start_idx:end_idx]

                if len(window) < self.window_size:
                    continue

                try:
                    telemetry = self.create_time_series_features(window)
                    vehicle_context = self.create_vehicle_context(window, static_info)
                    baseline_mpg = static_info.get('EPA Combined MPG', 28.0)

                    windows.append({
                        'telemetry': telemetry,
                        'vehicle_context': vehicle_context,
                        'fuel_loss': self.calculate_fuel_loss(window, baseline_mpg),
                        'behavior_class': self.classify_behavior(window),
                        'route_efficiency': self.calculate_route_efficiency(window),
                        'trip_id': trip_id,
                        'vehicle_id': int(signals['VehId'].iloc[0]),
                    })
                except Exception as e:
                    self.stats['skipped_errors'] += 1
                    continue

        return windows

    def prepare_dataset(self, signals_path: str, static_path: str,
                        vehicle_ids: list) -> Dict:
        all_windows = []

        print(f"  Processing {len(vehicle_ids)} vehicles...")

        for vehicle_id in vehicle_ids:
            try:
                signals, static_info = self.load_ved_data(
                    signals_path, static_path, vehicle_id
                )
                n_trips = signals['Trip'].nunique()
                n_rows = len(signals)

                windows = self.create_windows(signals, static_info)
                all_windows.extend(windows)

                self.stats['per_vehicle'][int(vehicle_id)] = {
                    'windows': len(windows),
                    'trips': n_trips,
                    'rows': n_rows,
                }
                print(f"    Vehicle {vehicle_id}: {len(windows)} windows "
                      f"({n_trips} trips, {n_rows} rows)")
            except Exception as e:
                print(f"    Vehicle {vehicle_id}: ERROR — {e}")
                continue

        if len(all_windows) == 0:
            raise RuntimeError(
                "No windows created from any vehicle. "
                "Check vehicle_ids and data file paths."
            )

        self.stats['total_windows'] = len(all_windows)

        telemetry = np.array([w['telemetry'] for w in all_windows])
        vehicle_context = np.array([w['vehicle_context'] for w in all_windows])
        fuel_loss = np.array([w['fuel_loss'] for w in all_windows], dtype=np.float32)
        behavior_class = np.array([w['behavior_class'] for w in all_windows], dtype=np.int64)
        route_efficiency = np.array([w['route_efficiency'] for w in all_windows], dtype=np.float32)
        vehicle_ids_arr = np.array([w['vehicle_id'] for w in all_windows])
        trip_ids_arr = np.array([w['trip_id'] for w in all_windows])

        vehicle_context = np.nan_to_num(vehicle_context, nan=0.0)
        telemetry = np.nan_to_num(telemetry, nan=0.0, posinf=0.0, neginf=0.0)

        batch_size, C, T = telemetry.shape
        telemetry_normalized = telemetry.copy()
        for c in range(C):
            channel_data = telemetry[:, c, :].flatten()
            mean_val = float(channel_data.mean())
            std_val = float(channel_data.std()) + 1e-8
            telemetry_normalized[:, c, :] = (telemetry[:, c, :] - mean_val) / std_val
            self.telemetry_scalers[c] = {'mean': mean_val, 'std': std_val}

        vehicle_context_scaled = self.scaler.fit_transform(vehicle_context)

        beh_dist = np.bincount(behavior_class, minlength=6)
        labels = ['Eco', 'Moderate', 'AggrAccel', 'HarshBrake', 'HighRPM', 'StopGo']
        print(f"\n  Behavior distribution:")
        for i, (label, count) in enumerate(zip(labels, beh_dist)):
            pct = count / len(behavior_class) * 100
            bar = '█' * int(pct / 2)
            print(f"    [{i}] {label:12s}: {count:4d} ({pct:5.1f}%) {bar}")

        n_trips = len(set(w['trip_id'] for w in all_windows))

        return {
            'telemetry': telemetry_normalized.astype(np.float32),
            'telemetry_raw': telemetry.astype(np.float32),
            'vehicle_context': vehicle_context_scaled.astype(np.float32),
            'vehicle_context_raw': vehicle_context.astype(np.float32),
            'fuel_loss': fuel_loss,
            'behavior_class': behavior_class,
            'route_efficiency': route_efficiency,
            'vehicle_ids': vehicle_ids_arr,
            'trip_ids': trip_ids_arr,
            'metadata': {
                'n_samples': len(all_windows),
                'n_vehicles': len(set(int(w['vehicle_id']) for w in all_windows)),
                'n_trips': n_trips,
                'telemetry_scalers': self.telemetry_scalers,
                'stats': self.stats,
            }
        }