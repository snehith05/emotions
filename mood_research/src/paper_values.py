"""
Every correlation reported in Wang et al. (2014), Tables 3 and 5-9, copied verbatim from the paper,
mapped to this project's feature and outcome names.
(table, paper wording, feature, outcome, r, p)
"""
PAPER = [
    # Table 3 - automatic sensing vs PHQ-9 depression
    ("3", "sleep duration (pre)", "sleep_h", "phq9_pre", -0.360, 0.025),
    ("3", "sleep duration (post)", "sleep_h", "phq9_post", -0.382, 0.020),
    ("3", "conversation frequency during day (pre)", "conv_freq_day", "phq9_pre", -0.403, 0.010),
    ("3", "conversation frequency during day (post)", "conv_freq_day", "phq9_post", -0.387, 0.016),
    ("3", "conversation frequency during evening (post)", "conv_freq_evening", "phq9_post", -0.345, 0.034),
    ("3", "conversation duration during day (post)", "conv_min_day", "phq9_post", -0.328, 0.044),
    ("3", "number of co-locations (post)", "colocations", "phq9_post", -0.362, 0.025),
    # Table 5 - flourishing
    ("5", "conversation duration (pre)", "conv_min", "flourishing_pre", 0.294, 0.066),
    ("5", "conversation duration during evening (pre)", "conv_min_evening", "flourishing_pre", 0.362, 0.022),
    ("5", "number of co-locations (post)", "colocations", "flourishing_post", 0.324, 0.050),
    # Table 6 - perceived stress scale
    ("6", "conversation duration (post)", "conv_min", "pss_post", -0.357, 0.026),
    ("6", "conversation frequency (post)", "conv_freq", "pss_post", -0.394, 0.013),
    ("6", "conversation duration during day (post)", "conv_min_day", "pss_post", -0.401, 0.011),
    ("6", "conversation frequency during day (pre)", "conv_freq_day", "pss_pre", -0.524, 0.001),
    ("6", "conversation frequency during evening (pre)", "conv_freq_evening", "pss_pre", -0.386, 0.015),
    ("6", "sleep duration (pre)", "sleep_h", "pss_pre", -0.355, 0.024),
    # Table 7 - loneliness
    ("7", "activity duration (post)", "activity_min", "loneliness_post", -0.388, 0.018),
    ("7", "activity duration for day (post)", "activity_min_day", "loneliness_post", -0.326, 0.049),
    ("7", "activity duration for evening (post)", "activity_min_evening", "loneliness_post", -0.464, 0.004),
    ("7", "traveled distance (post)", "distance_km", "loneliness_post", -0.338, 0.044),
    ("7", "traveled distance for day (post)", "distance_km_day", "loneliness_post", -0.336, 0.042),
    ("7", "indoor mobility for day (post)", "indoor_mobility_min_day", "loneliness_post", -0.332, 0.045),
    # Table 8 - EMA vs mental health
    ("8", "flourishing (pre) ~ positive affect", "pa", "flourishing_pre", 0.470, 0.002),
    ("8", "loneliness (post) ~ positive affect", "pa", "loneliness_post", -0.390, 0.020),
    ("8", "loneliness (post) ~ stress", "stress", "loneliness_post", 0.344, 0.037),
    ("8", "PHQ-9 (post) ~ stress", "stress", "phq9_post", 0.412, 0.010),
    ("8", "PSS (pre) ~ positive affect", "pa", "pss_pre", -0.387, 0.012),
    ("8", "PSS (post) ~ positive affect", "pa", "pss_post", -0.373, 0.019),
    ("8", "PSS (pre) ~ stress", "stress", "pss_pre", 0.458, 0.003),
    ("8", "PSS (post) ~ stress", "stress", "pss_post", 0.412, 0.009),
    # Table 9 - academic performance
    ("9", "spring GPA ~ conversation duration (day)", "conv_min_day", "gpa_spring", 0.356, 0.033),
    ("9", "spring GPA ~ conversation frequency (day)", "conv_freq_day", "gpa_spring", 0.334, 0.046),
    ("9", "spring GPA ~ indoor mobility", "indoor_mobility_min", "gpa_spring", -0.361, 0.031),
    ("9", "spring GPA ~ indoor mobility (day)", "indoor_mobility_min_day", "gpa_spring", -0.352, 0.036),
    ("9", "spring GPA ~ indoor mobility (night)", "indoor_mobility_min_night", "gpa_spring", -0.359, 0.032),
    ("9", "overall GPA ~ activity duration", "activity_min", "gpa_overall", -0.360, 0.030),
    ("9", "overall GPA ~ activity duration std deviation", "activity_min_sd", "gpa_overall", -0.479, 0.004),
    ("9", "overall GPA ~ indoor mobility", "indoor_mobility_min", "gpa_overall", -0.413, 0.014),
    ("9", "overall GPA ~ indoor mobility (day)", "indoor_mobility_min_day", "gpa_overall", -0.376, 0.026),
    ("9", "overall GPA ~ indoor mobility (night)", "indoor_mobility_min_night", "gpa_overall", -0.508, 0.002),
    ("9", "overall GPA ~ number of co-locations", "colocations", "gpa_overall", 0.447, 0.013),
]

# Table 4 - (n, mean, sd) pre and post
TABLE4 = {"phq9": (40, 5.8, 4.9, 38, 6.3, 5.8), "flourishing": (40, 42.6, 7.9, 37, 42.8, 8.9),
          "pss": (41, 18.4, 6.8, 39, 18.9, 7.1), "loneliness": (40, 40.5, 10.9, 37, 40.9, 10.5)}

# the sleep classifier's reported accuracy: +/- 32 min against ground truth (Jawbone UP, 10 students)
SLEEP_ACCURACY_MIN = 32
