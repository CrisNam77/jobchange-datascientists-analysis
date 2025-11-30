import numpy as np

NA_VALUES = [
    "",
    " ",
    "NA",
    "NaN",
    "nan",
    "NULL",
    "Unknown",
    "unknown",
    "Other",
    "others",
]


def load_csv_numpy(file_path):
    data = np.genfromtxt(file_path, delimiter=",", dtype=str, encoding="utf-8")

    col_names = data[0, :]

    data = data[1:]

    return col_names, data


def col_idx(col_names, col_name):
    col_names = np.array(col_names).astype(str)
    # Nếu là mảng 2D thì flatten
    if col_names.ndim > 1:
        col_names = col_names.flatten()

    matches = np.where(col_names == col_name)[0]
    if matches.size == 0:
        raise ValueError(
            f"Không tìm thấy cột '{col_name}' trong header.\nHeader thực tế: {col_names}"
        )
    return int(matches[0])


def to_nan(arr_str):
    arr = arr_str.copy().astype(object)
    mask_na = np.isin(arr, NA_VALUES)
    arr[mask_na] = np.nan
    return arr


def encode_ordinal_experience(exp_col):

    exp = to_nan(exp_col).astype(object)
    out = np.empty(exp.shape[0], dtype=float)
    out[:] = np.nan

    mask_lt1 = exp == "<1"
    mask_gt20 = exp == ">20"

    # giá trị số bình thường
    mask_digit = ~(mask_lt1 | mask_gt20)
    mask_digit = mask_digit & np.array([x is not np.nan for x in exp])

    out[mask_lt1] = 0.5
    out[mask_gt20] = 21.0
    out[mask_digit] = np.array(
        [float(x) if x is not np.nan else np.nan for x in exp[mask_digit]]
    )
    return out


def encode_ordinal_last_new_job(col):
    # last_new_job: 'never','1','2','3','4','>4'
    # map 'never' -> 0, '>4' -> 5, còn lại giữ số.

    arr = to_nan(col).astype(object)
    out = np.empty(arr.shape[0], dtype=float)
    out[:] = np.nan

    for i, v in enumerate(arr):
        if v is np.nan:
            out[i] = np.nan
        elif v == "never":
            out[i] = 0.0
        elif v == ">4":
            out[i] = 5.0
        else:
            out[i] = float(v)
    return out


def encode_ordinal_company_size(col):
    # company_size: '10-49','50-99','100-500','>1000', v.v.
    # map về center point gần đúng.
    arr = to_nan(col).astype(object)
    out = np.empty(arr.shape[0], dtype=float)
    out[:] = np.nan

    for i, v in enumerate(arr):
        if v is np.nan:
            out[i] = np.nan
            continue
        if "-" in v:
            a, b = v.split("-")
            out[i] = (float(a) + float(b)) / 2.0
        elif v.startswith(">"):
            num = float(v[1:])
            out[i] = num * 1.2  # giả sử >1000 ~ 1200
        elif v.startswith("<"):
            num = float(v[1:])
            out[i] = num / 2.0
        else:
            try:
                out[i] = float(v)
            except ValueError:
                out[i] = np.nan
    return out


def encode_categorical(col):
    # copy sang mảng object
    arr = col.copy().astype(object)

    # đánh dấu missing
    mask_na = np.isin(arr, NA_VALUES)
    arr[mask_na] = "__MISSING__"

    uniq, inv = np.unique(arr.astype(str), return_inverse=True)
    mapping = {val: i for i, val in enumerate(uniq)}
    return inv.astype(int), mapping


def fill_nan_with_stat(x, strategy="mean"):
    # Điền missing values cho 1 vector numeric x (float) theo mean/median.
    x = x.astype(float)
    mask_nan = np.isnan(x)
    if strategy == "mean":
        val = np.nanmean(x)
    elif strategy == "median":
        val = np.nanmedian(x)
    else:
        raise ValueError("Unknown strategy")

    x[mask_nan] = val
    return x, val


def standardize_column(x):
    # Z-score: (x - mean) / std, bỏ qua nan khi tính.
    mean = np.nanmean(x)
    std = np.nanstd(x)
    if std == 0:
        std = 1.0
    z = (x - mean) / std
    return z, mean, std


def min_max_scale(x):
    # Min-max [0,1]
    mn = np.nanmin(x)
    mx = np.nanmax(x)
    if mx == mn:
        return np.zeros_like(x), mn, mx
    scaled = (x - mn) / (mx - mn)
    return scaled, mn, mx


def preprocess_train(train_path, save_csv_path=None):
    # Tiền xử lý toàn bộ aug_train.csv:
    # - Bỏ enrollee_id (không dùng làm feature)
    # - Encode ordinal & nominal
    # - Xử lý missing (Option A: Unknown -> missing và fill)
    # - Chuẩn hoá numeric (z-score, log1p cho training_hours)

    # Trả về:
    #     X: features (2D float)
    #     y: target (1D int)
    #     meta: dict chứa info để preprocess test cho đồng nhất

    header, data_str = load_csv_numpy(train_path)
    h = header
    col = lambda name: data_str[:, col_idx(h, name)]

    # các cột
    # enrollee_id = col("enrollee_id")  # không dùng cho model
    city = col("city")
    city_dev = col("city_development_index")
    gender = col("gender")
    rel_exp = col("relevent_experience")
    enrolled_uni = col("enrolled_university")
    edu_level = col("education_level")
    major_disc = col("major_discipline")
    experience = col("experience")
    company_size = col("company_size")
    company_type = col("company_type")
    last_new_job = col("last_new_job")
    training_hours = col("training_hours")
    target = col("target").astype(float).astype(int)

    # numeric gốc
    city_dev_num = to_nan(city_dev).astype(float)
    training_hours_num = to_nan(training_hours).astype(float)

    # ordinal
    exp_num = encode_ordinal_experience(experience)
    comp_size_num = encode_ordinal_company_size(company_size)
    last_new_job_num = encode_ordinal_last_new_job(last_new_job)

    # categorical nominal encode
    gender_code, gender_map = encode_categorical(gender)
    rel_exp_code, rel_exp_map = encode_categorical(rel_exp)
    enrolled_uni_code, enrolled_uni_map = encode_categorical(enrolled_uni)
    edu_level_code, edu_level_map = encode_categorical(edu_level)
    major_disc_code, major_disc_map = encode_categorical(major_disc)
    company_type_code, company_type_map = encode_categorical(company_type)
    city_code, city_map = encode_categorical(city)

    # fill missing numeric
    city_dev_num, city_dev_fill = fill_nan_with_stat(city_dev_num, "mean")
    training_hours_num, training_hours_fill = fill_nan_with_stat(
        training_hours_num, "median"
    )
    exp_num, exp_fill = fill_nan_with_stat(exp_num, "median")
    comp_size_num, comp_size_fill = fill_nan_with_stat(comp_size_num, "median")
    last_new_job_num, last_new_job_fill = fill_nan_with_stat(last_new_job_num, "median")

    # chuẩn hoá numeric
    city_dev_z, city_dev_mean, city_dev_std = standardize_column(city_dev_num)
    tr_hours_log = np.log1p(training_hours_num)
    tr_hours_z, tr_hours_mean, tr_hours_std = standardize_column(tr_hours_log)
    exp_z, exp_mean, exp_std = standardize_column(exp_num)
    comp_size_z, comp_size_mean, comp_size_std = standardize_column(comp_size_num)
    last_new_job_z, last_mean, last_std = standardize_column(last_new_job_num)

    feature_names = [
        "city_dev_z",
        "training_hours_z",
        "experience_z",
        "company_size_z",
        "last_new_job_z",
        "city_code",
        "gender_code",
        "rel_exp_code",
        "enrolled_uni_code",
        "edu_level_code",
        "major_disc_code",
        "company_type_code",
    ]

    X = np.column_stack(
        [
            city_dev_z,
            tr_hours_z,
            exp_z,
            comp_size_z,
            last_new_job_z,
            city_code,
            gender_code,
            rel_exp_code,
            enrolled_uni_code,
            edu_level_code,
            major_disc_code,
            company_type_code,
        ]
    )

    y = target

    meta = {
        "header": header,
        "feature_names": feature_names,
        "city_map": city_map,
        "gender_map": gender_map,
        "rel_exp_map": rel_exp_map,
        "enrolled_uni_map": enrolled_uni_map,
        "edu_level_map": edu_level_map,
        "major_disc_map": major_disc_map,
        "company_type_map": company_type_map,
        "fill_values": {
            "city_dev": city_dev_fill,
            "training_hours": training_hours_fill,
            "experience": exp_fill,
            "company_size": comp_size_fill,
            "last_new_job": last_new_job_fill,
        },
        "scaling": {
            "city_dev": (city_dev_mean, city_dev_std),
            "training_hours_log": (tr_hours_mean, tr_hours_std),
            "experience": (exp_mean, exp_std),
            "company_size": (comp_size_mean, comp_size_std),
            "last_new_job": (last_mean, last_std),
        },
    }

    if save_csv_path is not None:
        data_proc = np.column_stack([y.astype(float), X.astype(float)])
        header_line = "target," + ",".join(feature_names)
        np.savetxt(
            save_csv_path,
            data_proc,
            delimiter=",",
            header=header_line,
            comments="",
            fmt="%.6f",
        )

    return X, y, meta


def preprocess_test(test_path, meta, save_csv_path=None):
    header, data_str = load_csv_numpy(test_path)
    h = header

    col = lambda name: data_str[:, col_idx(h, name)]

    enrollee_id = col("enrollee_id")
    city = col("city")
    city_dev = col("city_development_index")
    gender = col("gender")
    rel_exp = col("relevent_experience")
    enrolled_uni = col("enrolled_university")
    edu_level = col("education_level")
    major_disc = col("major_discipline")
    experience = col("experience")
    company_size = col("company_size")
    company_type = col("company_type")
    last_new_job = col("last_new_job")
    training_hours = col("training_hours")

    city_dev_num = to_nan(city_dev).astype(float)
    training_hours_num = to_nan(training_hours).astype(float)
    exp_num = encode_ordinal_experience(experience)
    comp_size_num = encode_ordinal_company_size(company_size)
    last_new_job_num = encode_ordinal_last_new_job(last_new_job)

    fv = meta["fill_values"]
    city_dev_num[np.isnan(city_dev_num)] = fv["city_dev"]
    training_hours_num[np.isnan(training_hours_num)] = fv["training_hours"]
    exp_num[np.isnan(exp_num)] = fv["experience"]
    comp_size_num[np.isnan(comp_size_num)] = fv["company_size"]
    last_new_job_num[np.isnan(last_new_job_num)] = fv["last_new_job"]

    sc = meta["scaling"]

    def z_with_param(x, mean, std):
        if std == 0:
            std = 1.0
        return (x - mean) / std

    city_dev_z = z_with_param(city_dev_num, *sc["city_dev"])
    tr_hours_log = np.log1p(training_hours_num)
    tr_hours_z = z_with_param(tr_hours_log, *sc["training_hours_log"])
    exp_z = z_with_param(exp_num, *sc["experience"])
    comp_size_z = z_with_param(comp_size_num, *sc["company_size"])
    last_new_job_z = z_with_param(last_new_job_num, *sc["last_new_job"])

    def encode_with_map(col_str, mapping):
        arr = to_nan(col_str)
        codes = np.empty(arr.shape[0], dtype=int)
        for i, v in enumerate(arr):
            if v in mapping:
                codes[i] = mapping[v]
            else:
                codes[i] = -1
        return codes

    city_code = encode_with_map(city, meta["city_map"])
    gender_code = encode_with_map(gender, meta["gender_map"])
    rel_exp_code = encode_with_map(rel_exp, meta["rel_exp_map"])
    enrolled_uni_code = encode_with_map(enrolled_uni, meta["enrolled_uni_map"])
    edu_level_code = encode_with_map(edu_level, meta["edu_level_map"])
    major_disc_code = encode_with_map(major_disc, meta["major_disc_map"])
    company_type_code = encode_with_map(company_type, meta["company_type_map"])

    X = np.column_stack(
        [
            city_dev_z,
            tr_hours_z,
            exp_z,
            comp_size_z,
            last_new_job_z,
            city_code,
            gender_code,
            rel_exp_code,
            enrolled_uni_code,
            edu_level_code,
            major_disc_code,
            company_type_code,
        ]
    )

    feature_names = meta["feature_names"]

    if save_csv_path is not None:
        data_proc = np.column_stack([enrollee_id.astype(float), X.astype(float)])
        header_line = "enrollee_id," + ",".join(feature_names)
        np.savetxt(
            save_csv_path,
            data_proc,
            delimiter=",",
            header=header_line,
            comments="",
            fmt="%.6f",
        )

    return X, enrollee_id


def load_processed_train_csv(path):
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    y = data[:, 0].astype(int)
    X = data[:, 1:]
    return X, y


def load_processed_test_csv(path):
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    enrollee_id = data[:, 0].astype(int)
    X = data[:, 1:]
    return X, enrollee_id
