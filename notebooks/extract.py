import measurements
import astropy.time as astime


def features(df, num_features):
    '''
    Given a light curve as DataFrame, return all features in dict.
    '''
    df = df.copy()
    df['Flux'] = measurements.__mag_to_flux__(df.Mag)
    df['Date'] = astime.Time(df.MJD, format='mjd').datetime
    df = df.sort_values('Date')
    feature_dict = dict()
    # Curve fitting params
    (p1_params, p2_params, p3_params, p4_params) = measurements.poly_params(
        df.Mag, df.Magerr, df.MJD)
    feature_dict['poly1_t1'] = p1_params[0]
    feature_dict['poly2_t2'] = p2_params[0]
    feature_dict['poly2_t1'] = p2_params[1]
    feature_dict['poly3_t3'] = p3_params[0]
    feature_dict['poly3_t2'] = p3_params[1]
    feature_dict['poly3_t1'] = p3_params[2]
    feature_dict['poly4_t4'] = p4_params[0]
    feature_dict['poly4_t3'] = p4_params[1]
    feature_dict['poly4_t2'] = p4_params[2]
    feature_dict['poly4_t1'] = p4_params[3]
    # Curve statistics measurements
    feature_dict['skew'] = measurements.skew(df.Mag)
    feature_dict['kurtosis'] = measurements.kurtosis(df.Mag)
    feature_dict['small_kurtosis'] = measurements.small_kurtosis(df.Mag)
    feature_dict['std'] = measurements.std(df.Mag)
    feature_dict['beyond1st'] = measurements.beyond1st(df.Mag, df.Magerr)
    feature_dict['stetson_j'] = measurements.stetson_j(
        df.Mag, df.Magerr, df.Date)
    feature_dict['stetson_k'] = measurements.stetson_k(df.Mag, df.Magerr)
    feature_dict['max_slope'] = measurements.max_slope(df.Mag, df.Date)
    feature_dict['amplitude'] = measurements.amplitude(df.Mag)
    feature_dict['median_absolute_deviation'] = measurements.median_absolute_deviation(
        df.Mag)
    feature_dict['median_buffer_range_percentage'] = measurements.median_buffer_range_percentage(
        df.Flux)
    feature_dict['pair_slope_trend'] = measurements.pair_slope_trend(
        df.Mag, df.Date)
    feature_dict['pair_slope_trend_last_30'] = measurements.pair_slope_trend_last_30(
        df.Mag, df.Date)
    feature_dict['flux_percentile_ratio_mid20'] = measurements.flux_percentile_ratio_mid20(
        df.Flux)
    feature_dict['flux_percentile_ratio_mid35'] = measurements.flux_percentile_ratio_mid35(
        df.Flux)
    feature_dict['flux_percentile_ratio_mid50'] = measurements.flux_percentile_ratio_mid50(
        df.Flux)
    feature_dict['flux_percentile_ratio_mid65'] = measurements.flux_percentile_ratio_mid65(
        df.Flux)
    feature_dict['flux_percentile_ratio_mid80'] = measurements.flux_percentile_ratio_mid80(
        df.Flux)
    feature_dict['percent_amplitude'] = measurements.percent_amplitude(df.Flux)
    feature_dict['percent_difference_flux_percentile'] = measurements.percent_difference_flux_percentile(
        df.Flux)
#     feature_dict['linear_trend'] = measurements.linear_trend(df.Flux, df.Date)
    return feature_dict


def feature_dict(num_features=21):
    features = [
        'skew', 'std', 'kurtosis', 'beyond1st', 'stetson_j', 'stetson_k', 'max_slope',
        'amplitude', 'median_absolute_deviation', 'median_buffer_range_percentage',
        'pair_slope_trend', 'percent_amplitude', 'percent_difference_flux_percentile',
        'flux_percentile_ratio_mid20',  'flux_percentile_ratio_mid35',
        'flux_percentile_ratio_mid50', 'flux_percentile_ratio_mid65',
        'flux_percentile_ratio_mid80',
        'small_kurtosis', 'pair_slope_trend_last_30',
        #         'linear_trend'
    ]
    if num_features > 20:
        features.extend(['poly1_t1', 'poly2_t2', 'poly2_t1',
                         'poly3_t3', 'poly3_t2', 'poly3_t1'])
    if num_features > 26:
        features.extend(['poly4_t4', 'poly4_t3', 'poly4_t2', 'poly4_t1'])
    return {k: [] for k in features}
