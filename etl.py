"""
Extract, Transform and Load

이 모듈의 주요 역할
1. rawdata폴더에 있는 excel, csv파일들 전처리 진행
    - excel에 기입된 swell 데이터 정리 (raw_swell_data_processing())
        rawdata/problem.xlsx, result_sample.xlsx -> data/swell.csv (raw_feature_data_processing())
    - 연도별로 나뉘어있는 csv 파일들을 종류별로 묶고 index를 통일
    - 뉴럴넷에 넣기 좋게 dataset 생성
2. 각종 데이터 전처리
"""
import numpy as np
import pandas as pd
import datetime as dt
import h5py
import re
import os

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from constants import RAWDATA_PATH, DATA_PATH, NP_SEED, OBV_TYPES
np.random.seed(NP_SEED)

class ETL:

    def __init__(self, n_lookback, split_ratio, downsampling, reload):
        if not os.path.exists(DATA_PATH):
            reload = True
            os.mkdir(DATA_PATH)

        if reload:
            print('Load dataset from raw data...')
            self.raw_swell_data_processing()
            self.raw_feature_data_processing()
            self.df, self.feature_size_list = self.df_generating()
            self.ds = self.dataset_generating(self.df, n_lookback, split_ratio, downsampling)
        else:
            print('Load dataset from existing dataset')
            self.df, self.feature_size_list = self.df_generating()

        self.ds = self.read_from_h5()

        # 모든 데이터의 shape 출력
        print()
        for d0 in self.ds:
            for d1 in self.ds[d0]:
                print('{:>7} {:>3} {}'.format(d0, d1, self.ds[d0][d1].shape))

        # train, val, test 데이터의 분포 출력
        print()
        for d0, d0_dict in self.ds.items():
            if d0 == 'problem':
                continue
            print('[{} data 분포]'.format(d0))
            print('0 : 1 : 2 = {:.2} : {:.2} : {:.2}'.format((d0_dict['y'] == 0).sum() / len(d0_dict['y']),
                                                             (d0_dict['y'] == 1).sum() / len(d0_dict['y']),
                                                             (d0_dict['y'] == 2).sum() / len(d0_dict['y'])))

    @staticmethod
    def raw_swell_data_processing():
        """
        swell이 발생한 시간이 기입된 excel파일을 읽어들여 다루기 쉬운 형태의 데이터로 변경하여 .csv로 저장한다.
        결과 data/swell.csv 파일에는 2014-01-01 부터 2017-12-31까지 시간별로 아래의 데이터로 레이블링 된다.

        0: 기상 양호
        1: 기상 불량
        2: Swell
        Nan: posco에서 문제로 제시한 시간대

        :output: DATA_PATH/swell.csv
        """
        dt_index = pd.date_range('2014-01-01 07:00:00', '2017-12-31 23:00:00', freq='H')
        swell_df = pd.DataFrame(data={'swell': [0] * len(dt_index)}, index=dt_index)
        swell_df.index.name = 'date'

        BAD_WEATHER = 1
        SWELL = 2

        for year in ['2014', '2015', '2016', '2017']:
            raw_year_df = pd.read_excel(os.path.join(RAWDATA_PATH,'problem.xlsx'), sheet_name=year)
            raw_year_df.rename(columns={k: int(k.split()[-1]) for k in raw_year_df.columns.tolist()},
                               inplace=True)
            # 날짜가 적힌 columns들 만 fillna(method=ffill)
            raw_year_df.loc[:, list(range(0, 48, 4))] = raw_year_df.loc[:, list(range(0, 48, 4))].fillna(method='ffill')

            year_df = pd.DataFrame()
            for i in range(12):
                year_df = year_df.append(raw_year_df.loc[:, 4 * i:4 * i + 3].rename(
                    columns={4 * i: 0, 4 * i + 1: 1, 4 * i + 2: 2, 4 * i + 3: 3}))
            year_df.drop(columns=[3], inplace=True)
            year_df = year_df[~year_df.loc[:, 0].str.contains('계').fillna(False)]
            year_df = year_df[~year_df.loc[:, 0].str.contains('월').fillna(False)]
            year_df = year_df[~year_df.loc[:, 0].str.contains('일자').fillna(False)]
            year_df = year_df.fillna(0)
            year_df = year_df[year_df.loc[:, 2].str.contains('~').fillna(False)]

            for row in year_df.iterrows():
                date, swell, time = row[1][0], row[1][1], row[1][2]
                time = time.replace('24', '00')
                time_from, time_to = time.split('~')
                time_from = dt.datetime.strptime(time_from, "%H:%M").time()
                time_to = dt.datetime.strptime(time_to, "%H:%M").time()

                dt_from = dt.datetime(date.year, date.month, date.day, time_from.hour, time_from.minute)
                dt_to = dt.datetime(date.year, date.month, date.day, time_to.hour, time_to.minute)

                if dt_from.hour < 7:
                    dt_from = dt_from + dt.timedelta(days=1)

                if dt_from.minute == 30:
                    dt_from = dt_from - dt.timedelta(minutes=30)
                if dt_to.minute == 30:
                    dt_to = dt_to + dt.timedelta(minutes=30)

                dt_to = dt_to - dt.timedelta(hours=1)

                if dt_from > dt_to:
                    dt_to = dt_to + dt.timedelta(days=1)

                # 2014년 2월 14일 데이터와 같이 동 시간대에 Swell과 기상불량이 모두 기록되어있는 경우
                # SWELL로 간주한다. (데이터 레이블링 오류로 보여짐)
                for dt_idx in pd.date_range(dt_from, dt_to, freq='H'):
                    if swell:
                        swell_df.loc[dt_idx] = SWELL
                    elif swell_df.loc[dt_idx].swell != SWELL:
                        swell_df.loc[dt_idx] = BAD_WEATHER

        result_sample_df = pd.read_excel(os.path.join(RAWDATA_PATH, 'result_sample.xlsx'), sheet_name='Sheet1')
        test_date_list = result_sample_df[result_sample_df.columns[0]][2:].tolist()
        for dt_test in test_date_list:
            dt_from = dt_test + dt.timedelta(hours=7)
            dt_to = dt_test + dt.timedelta(hours=30)
            swell_df[dt_from:dt_to] = np.nan

        swell_df.to_csv(os.path.join(DATA_PATH, 'swell.csv'))

    @staticmethod
    def raw_feature_data_processing():

        csv_name_dict = get_csv_name_dict()

        # 연도별 데이터(14년, 15년, 16년, 17년)를 하나로 합쳐서 저장
        for code, csv_names in csv_name_dict.items():
            obv_type = re.match('[^_]+_[^_]+', code).group(0)
            dfs = []
            for csv_name in csv_names:
                df = pd.read_csv(os.path.join(RAWDATA_PATH, obv_type, csv_name),
                                 encoding='cp949')
                if obv_type == 'MARINE_BUOY' and csv_name.split('_')[5] == '2014':
                    df = df.rename(columns={'풍속1(m/s)': '풍속(m/s)',
                                            '풍향1(deg)': '풍향(deg)',
                                            'GUST풍속 1(m/s)': 'GUST풍속(m/s)'})
                dfs.append(df)
            df = pd.concat(dfs, join='outer', sort=False)
            df.to_csv(os.path.join(DATA_PATH, code + '.csv'), index=False)

        # index 통일하기
        dt_index = pd.date_range(start='2014-01-01 07:00:00', end='2017-12-31 23:00:00', freq='H')
        df_dict = {}
        for code in csv_name_dict.keys():
            df_dict[code] = pd.read_csv(os.path.join(DATA_PATH, code + '.csv'), index_col=1)
            df_dict[code].index = pd.to_datetime(df_dict[code].index)
            df_dict[code] = df_dict[code][~df_dict[code].index.duplicated(keep='first')]
            df_dict[code] = df_dict[code].reindex(index=dt_index)
            df_dict[code].index.name = 'date'
            df_dict[code].drop(columns=['지점'], inplace=True)
            df_dict[code].to_csv(os.path.join(DATA_PATH, code + '.csv'), index=True)

    @staticmethod
    def df_generating():
        """
        사용할 데이터를 모두 모아 하나의 pd.DataFrame으로 만든다.
        사용할 데이터는 다음과 같다.
        1. 'marine_buoy'   : 기온, 습도, 풍향, 풍속, GUST풍속, 수온, 현지기압, 최대파고, 유의파고, 평균파고, 파주기, 파향 (모든 컬럼임)
        2. 'marine_cwbuoy' : 수온, 최대파고, 유의파고, 평균파고, 파주기 (모든 컬럼임)
        3. 'marine_lh' : 모든 컬럼
        4. 'surface_asos'  : 기온, 습도, 풍향, 풍속, 강수량, 증기압, 이슬점, 현지기압, 해면기압, 지면온도
        5. 'surface_aws'   : 기온, 풍향, 풍속, 강수량
        6. t-1 시점의 swell 여부
        7. t 시점의 swell 여부
        머신러닝 관점에서 추후 1~6는 feature가 되며, 7은 y가 된다.
        """

        csv_name_dict = get_csv_name_dict()

        df_dict = {}
        for code in csv_name_dict.keys():
            df_dict[code] = pd.read_csv(os.path.join(DATA_PATH, code + '.csv'), index_col=0, parse_dates=True)

        # columns name에서 () 안의 내용 없애기
        for df in df_dict.values():
            columns = df.columns.tolist()
            columns = dict.fromkeys(columns)
            for col in columns:
                columns[col] = re.sub(r'\([^)]*\)', '', col)
            df.rename(columns=columns, inplace=True)

        # 사용할 column만 필터링
        for code, df in df_dict.items():
            obv_type = re.match('[^_]+_[^_]+', code).group(0)
            if obv_type == 'SURFACE_ASOS':
                df_dict[code] = df[['기온', '습도', '풍향', '풍속', '강수량',
                                    '증기압', '이슬점온도', '현지기압', '해면기압',
                                    '지면온도']]
            #                              '5cm 지중온도', '10cm 지중온도',
            #                              '20cm 지중온도', '30cm 지중온도']]

            elif obv_type == 'SURFACE_AWS':
                df_dict[code] = df[['기온', '풍향', '풍속', '강수량']]

            elif obv_type == 'MARINE_BUOY':
                df_dict[code] = df[['기온', '습도', '풍향', '풍속', '수온',
                                    '최대파고', '유의파고', '평균파고', '파주기',
                                    'GUST풍속', '현지기압', '파향']]

            elif obv_type == 'MARINE_CWBUOY':
                pass
            elif obv_type == 'MARINE_LH':
                pass

        # Handling missing data...
        dt_nan = pd.date_range(pd.datetime(2014, 1, 1, 7), pd.datetime(2015, 9, 25, 10), freq='H')
        df_dict['MARINE_CWBUOY_22490'].loc[dt_nan] = df_dict['MARINE_CWBUOY_22453'].loc[dt_nan]  # 월포 <- 구룡포

        dt_nan = pd.date_range(pd.datetime(2014, 1, 1, 7), pd.datetime(2015, 12, 8, 23), freq='H')
        df_dict['MARINE_BUOY_22190'].loc[dt_nan] = df_dict['MARINE_BUOY_22106'].loc[dt_nan]  # 울진 <- 포항

        # 강수량 -> fillna(0), 그 외 -> interpolate
        for code, df in df_dict.items():
            obv_type = re.match('[^_]+_[^_]+', code).group(0)
            if obv_type in ['SURFACE_ASOS', 'SURFACE_AWS']:
                df['강수량'].fillna(0, inplace=True)
            df.interpolate(inplace=True)

        # 풍향, 파향 데이터는 degree -> (sin(rad), cos(rad)) 로 변환한다.
        for code, df in df_dict.items():
            direction_cols = [x for x in df.columns.tolist() if '풍향' in x or '파향' in x]
            for col in direction_cols:
                df_dict[code][col+'sin'] = df_dict[code][col].apply(deg2sin)
                df_dict[code][col+'cos'] = df_dict[code][col].apply(deg2cos)
                del df_dict[code][col]

        feature_size_list = []  # CNN에서 사용될 수 있다.
        # column 이름에 csv파일 이름 붙이기
        for code, df in df_dict.items():
            col_dict = {k: code + ':' + k for k in df.columns}
            df_dict[code] = df.rename(columns=col_dict)

            feature_size_list.append(len(col_dict))

        # 모든 df 하나로 합치기
        df = pd.concat(df_dict.values(), axis=1, sort=False)

        # 풍향, 파향 데이터는 위에서 변환한 그대로 사용, 나머지는 standard로 scaling
        direction_cols = [x for x in df.columns.tolist() if '풍향' in x or '파향' in x]
        standard_cols = [x for x in df.columns.tolist() if x not in direction_cols]

        scaler = StandardScaler()
        scaler.fit(df[standard_cols])
        df[standard_cols] = scaler.transform(df[standard_cols])

        # t-1 시점의 swell 여부를 feature로 추가
        swell_df = pd.read_csv(os.path.join(DATA_PATH, 'swell.csv'), index_col=0, parse_dates=True)
        df = df[df.index >= pd.datetime(2014, 1, 1, 8, 0, 0)]
        df.loc[:, "swell_t-1"] = swell_df['swell'].tolist()[:-1]
        df.loc[:, "swell"] = swell_df['swell'].tolist()[1:]

        return df, feature_size_list

    @staticmethod
    def dataset_generating(df, n_lookback, split_ratio, downsampling):
        """
        뉴럴넷에 사용하기 위한 형태의 dataset으로 저장한다.
        뉴럴넷에 사용할 X와 y는 다음과 같다.
        X : 과거 N_LOOKBACK시간 데이터
        y : t~t+1 사이의 swell 발생 여부

        e.g. N_LOOKBACK=8이고 사용하는 feature 개수가 47개인 경우,
        X : t-7 t-6 t-5 t-4 t-3 t-2 t-1 t (shape=[8, 47])
        y : t시간에서 t+1시간 사이의 swell 발생 여부 (shape=[1])
        * y는 현재 swell 데이터의 index로 설명하면 t 시점의 swell 여부를 의미한다.
        * N_LOOKBACK은 24를 넘을 수 없다.

        :output: DATA_PATH/dataset.h5
        """

        ds = {'train': {}, 'val': {}, 'test': {}, 'problem': {}}

        truncated_df = df[(dt.datetime(2014, 1, 2, 7) <= df.index) * (df.index <= dt.datetime(2017, 12, 31, 6))]
        dt_problem = df[df.isnull().swell].index
        dt_problem = np.array(dt_problem).reshape((-1, 24))

        # 기상불량 또는 swell이 있었던 날들 (엑셀에 기재된 날들) 찾기
        dt_excel = []
        for i in range(len(truncated_df) // 24):
            swell_i = truncated_df.iloc[i * 24:(i + 1) * 24, -1]
            if swell_i.sum() != 0 and swell_i.sum() != np.nan:
                dt_excel.append(swell_i.index)
        print('엑셀에 기재된 날들: {} + {}(problem)'.format(len(dt_excel), len(dt_problem)))
        dt_excel_flat = np.array(dt_excel).ravel()
        swell_0 = len(df.loc[dt_excel_flat][df.loc[dt_excel_flat, 'swell'] == 0])
        swell_1 = len(df.loc[dt_excel_flat][df.loc[dt_excel_flat, 'swell'] == 1])
        swell_2 = len(df.loc[dt_excel_flat][df.loc[dt_excel_flat, 'swell'] == 2])
        swell_0_ratio = swell_0 / len(dt_excel_flat)
        swell_1_ratio = swell_1 / len(dt_excel_flat)
        swell_2_ratio = swell_2 / len(dt_excel_flat)
        print('0 : 1 : 2 = {:.2} : {:.2} : {:.2}'.format(swell_0_ratio, swell_1_ratio, swell_2_ratio))

        # Problem으로 주어진 날의 다음날 데이터는 Val과 Test에 사용하지 않는다
        def dt_in_dts(t, t_list):
            for _t in t_list:
                if np.prod(t == _t) == 1:
                    return True
            return False

        new_dt_excel = []
        for t in dt_excel:
            if not dt_in_dts(t - np.timedelta64(dt.timedelta(hours=24)), dt_problem):
                new_dt_excel.append(t)
        print('\ndt problem 날짜와 다음날 데이터를 제외한 엑셀에 기재된 날들 개수:', len(new_dt_excel))

        p = np.random.permutation(len(new_dt_excel))
        new_dt_excel = np.array(new_dt_excel)[p]
        train_idx = int(len(new_dt_excel) * split_ratio[0])
        val_idx = train_idx + int(len(new_dt_excel) * split_ratio[1])

        dt_val = new_dt_excel[train_idx:val_idx]
        dt_test = new_dt_excel[val_idx:]

        print('위 개수의 엑셀에 기재된 날들 중')
        print('val에 {}days는 test에 {}days 사용, 나머지는 train에 사용'.format(len(dt_val), len(dt_test)))

        ds['val']['dt'] = pd.DatetimeIndex(dt_val.ravel())
        ds['test']['dt'] = pd.DatetimeIndex(dt_test.ravel())
        ds['problem']['dt'] = pd.DatetimeIndex(dt_problem.ravel())
        dt_train = truncated_df.index
        ds['train']['dt'] = pd.DatetimeIndex(x for x in dt_train if x not in ds['val']['dt'] \
                                             and x not in ds['test']['dt'] \
                                             and x not in ds['problem']['dt'])


        def get_x_y(dt_list, get_y=True):

            x_list = []
            y_list = []
            for t in dt_list:
                x_idx = df.index.get_loc(t) + 1
                values = df.iloc[x_idx - n_lookback: x_idx].values
                x_list.append(values[:, :-1])
                y_list.append(values[-1, -1])
            x_list = np.array(x_list)
            y_list = np.array(y_list)[:, np.newaxis].astype(int)

            if get_y:
                return x_list, y_list
            else:
                return x_list

        ds['train']['x'], ds['train']['y'] = get_x_y(ds['train']['dt'])
        ds['val']['x'], ds['val']['y'] = get_x_y(ds['val']['dt'])
        ds['test']['x'], ds['test']['y'] = get_x_y(ds['test']['dt'])
        ds['problem']['x'] = get_x_y(ds['problem']['dt'], get_y=False)

        if downsampling:
            # 맞춰야 하는 날이 동안 0만 있는 경우, train 데이터에서 제외
            print('\n[Down Sampling 전의 train data 분포] 총 개수:', len(ds['train']['y']))
            print('0 : 1 : 2 = {:.2} : {:.2} : {:.2}'.format((ds['train']['y'] == 0).sum()/len(ds['train']['y']),
                                                             (ds['train']['y'] == 1).sum()/len(ds['train']['y']),
                                                             (ds['train']['y'] == 2).sum()/len(ds['train']['y'])))
            tmp = ds['train']['y'].reshape(-1, 24)
            idx = ~np.prod(tmp == 0, axis=1).astype(np.bool)
            idx = np.stack([idx] * 24, axis=1).ravel()
            ds['train']['x'], ds['train']['y'], ds['train']['dt'] = ds['train']['x'][idx], \
                                                                    ds['train']['y'][idx], \
                                                                    ds['train']['dt'][idx]

            print('[Down Sampling 후의 train data 분포] 총 개수:', len(ds['train']['y']))
            print('0 : 1 : 2 = {:.2} : {:.2} : {:.2}'.format((ds['train']['y'] == 0).sum()/len(ds['train']['y']),
                                                             (ds['train']['y'] == 1).sum()/len(ds['train']['y']),
                                                             (ds['train']['y'] == 2).sum()/len(ds['train']['y'])))

        # remove nan
        idx = ~np.max(np.isnan(ds['train']['x']), (1, 2))
        ds['train']['x'] = ds['train']['x'][idx]
        ds['train']['y'] = ds['train']['y'][idx]
        ds['train']['dt'] = ds['train']['dt'][idx]

        # Shuffle train data
        ds['train']['x'], ds['train']['y'], ds['train']['dt'] = random_shuffle(ds['train']['x'],
                                                                               ds['train']['y'],
                                                                               ds['train']['dt'])

        ds['train']['dt'] = np.array(ds['train']['dt'].strftime("%Y%m%d%H"), dtype='S')
        ds['val']['dt'] = np.array(ds['val']['dt'].strftime("%Y%m%d%H"), dtype='S')
        ds['test']['dt'] = np.array(ds['test']['dt'].strftime("%Y%m%d%H"), dtype='S')
        ds['problem']['dt'] = np.array(ds['problem']['dt'].strftime("%Y%m%d%H"), dtype='S')

        with h5py.File(os.path.join(DATA_PATH, 'dataset.h5'), 'w') as hf:
            hf.create_dataset('train_x', shape=ds['train']['x'].shape, data=ds['train']['x'])
            hf.create_dataset('train_y', shape=ds['train']['y'].shape, data=ds['train']['y'], dtype=int)
            hf.create_dataset('train_dt', shape=ds['train']['dt'].shape, dtype='S10', data=ds['train']['dt'])
            hf.create_dataset('val_x', shape=ds['val']['x'].shape, data=ds['val']['x'])
            hf.create_dataset('val_y', shape=ds['val']['y'].shape, data=ds['val']['y'], dtype=int)
            hf.create_dataset('val_dt', shape=ds['val']['dt'].shape, dtype='S10', data=ds['val']['dt'])
            hf.create_dataset('test_x', shape=ds['test']['x'].shape, data=ds['test']['x'])
            hf.create_dataset('test_y', shape=ds['test']['y'].shape, data=ds['test']['y'], dtype=int)
            hf.create_dataset('test_dt', shape=ds['test']['dt'].shape, dtype='S10', data=ds['test']['dt'])
            hf.create_dataset('problem_x', shape=ds['problem']['x'].shape, data=ds['problem']['x'])
            hf.create_dataset('problem_dt', shape=ds['problem']['dt'].shape, dtype='S10', data=ds['problem']['dt'])

        return ds


    @staticmethod
    def read_from_h5():

        ds = {'train': {}, 'val': {}, 'test': {}, 'problem': {}}
        with h5py.File(os.path.join(DATA_PATH, 'dataset.h5'), 'r') as hf:
            ds['train']['x'] = hf['train_x'][:]
            ds['train']['y'] = hf['train_y'][:]
            ds['train']['dt'] = hf['train_dt'][:].astype(str)
            ds['train']['dt'] = np.array([dt.datetime.strptime(x, "%Y%m%d%H") for x in ds['train']['dt']])
            ds['val']['x'] = hf['val_x'][:]
            ds['val']['y'] = hf['val_y'][:]
            ds['val']['dt'] = hf['val_dt'][:].astype(str)
            ds['val']['dt'] = np.array([dt.datetime.strptime(x, "%Y%m%d%H") for x in ds['val']['dt']])
            ds['test']['x'] = hf['test_x'][:]
            ds['test']['y'] = hf['test_y'][:]
            ds['test']['dt'] = hf['test_dt'][:].astype(str)
            ds['test']['dt'] = np.array([dt.datetime.strptime(x, "%Y%m%d%H") for x in ds['test']['dt']])
            ds['problem']['x'] = hf['problem_x'][:]
            ds['problem']['dt'] = hf['problem_dt'][:].astype(str)
            ds['problem']['dt'] = np.array([dt.datetime.strptime(x, "%Y%m%d%H") for x in ds['problem']['dt']])

        return ds

    def oversampling(self, times=2):
        """
        swell_t-1과 맞춰야하는 swell이 다른 샘플들을 오버샘플링
        마지막 feature가 swell_t-1 이어야 함
        times: 배수 e.g. 1이면 데이터셋 변함 없음. (int)
        """

        diff_samples = self.ds['train']['x'][:, -1, -1] != self.ds['train']['y'][:, -1]
        diff_samples_x = self.ds['train']['x'][diff_samples]
        diff_samples_y = self.ds['train']['y'][diff_samples]
        diff_samples_dt = self.ds['train']['dt'][diff_samples]

        for i in range(times-1):
            self.ds['train']['x'] = np.concatenate([self.ds['train']['x'], diff_samples_x], axis=0)
            self.ds['train']['y'] = np.concatenate([self.ds['train']['y'], diff_samples_y], axis=0)
            self.ds['train']['dt'] = np.concatenate([self.ds['train']['dt'], diff_samples_dt], axis=0)

        p = np.random.permutation(len(self.ds['train']['x']))
        self.ds['train']['x'] = self.ds['train']['x'][p]
        self.ds['train']['y'] = self.ds['train']['y'][p]
        self.ds['train']['dt'] = self.ds['train']['dt'][p]
        
        
def get_csv_name_dict():
    # key: value = MARINE_BUOY_22106: [MARINE_BUOY_22106_HR_2014_2014_2015.csv, ...]
    csv_name_dict = {}
    for obv_type in OBV_TYPES:
        obv_path = os.path.join(RAWDATA_PATH, obv_type)
        for csv_name in os.listdir(obv_path):
            code = re.match('[^_]+_[^_]+_[^_]+', csv_name).group(0)
            if code not in csv_name_dict.keys():
                csv_name_dict[code] = []
            csv_name_dict[code].append(csv_name)

    return csv_name_dict


def deg2sincos(degree):
    cos = np.cos(np.deg2rad(degree))
    sin = np.sin(np.deg2rad(degree))
    return sin, cos


def deg2sin(deg):
    return np.sin(np.deg2rad(deg))


def deg2cos(deg):
    return np.cos(np.deg2rad(deg))


def random_shuffle(x, y, aux):
    p = np.random.permutation(len(x))
    return x[p], y[p], aux[p]
