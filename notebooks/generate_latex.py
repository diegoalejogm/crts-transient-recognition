
from string import Template
import helpers as h
import exploration as e
import classification as c
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
import features

FIGURES_PATH = c.RESULTS_PATH + 'figures/'


def pretty_task_name(task_index):
    return ['Binary', '6-Transient',
            '7-Transient', '7-Class', '8-Class'][task_index]


def pretty_class_labels(labels):
    result = []
    for i in range(labels.shape[0]):
        label = labels[i]
        if label == 0:
            label = 'nontransient'
        elif label == 1:
            label = 'transient'
        result.append(label)
    return np.array(result)


def generate_latex_results():
    # Obtain Row for Top Classifier in Task
    top_functions = [e.top_binary, e.top_six_t,
                     e.top_seven_t, e.top_seven_c, e.top_eight_c]
    for task_index in range(len(top_functions)):
        task_name = pretty_task_name(task_index)
#         print(task_name)
        # Obtain Task DataFrame
        task_df = e._load_dataframe_(task_index)
#         print(task_index, task_df.max()['test_fscore'])
        # Obtain best classifier data for current task
        top_row, top_model = top_functions[task_index]()
#         print(top_row)
#         print(top_row)

        # Obtain Latex Tables
        cnf_matrix = task_confusion_matrix(task_index, top_row)
        scores_reg = task_scores_latex(task_index, top_row, False)
        class_reg = task_classifiers_score_latex(
            task_df, task_index, 'fscore', False)

        importances = importances_graph(top_model, task_index, 'figures/')

        if task_index > 0:
            scores_bal = task_scores_latex(task_index, top_row, True)
            class_bal = task_classifiers_score_latex(
                task_df, task_index, 'fscore', True)

        # Create Output Dir
        out_dir = FIGURES_PATH + task_name + '/'
        h.make_dir_if_not_exists(out_dir)
        # Save latex files
        with open(out_dir + 'cnf_matrix.tex', 'w') as f:
            f.write(cnf_matrix)
        with open(out_dir + 'scores_reg.tex', 'w') as f:
            f.write(scores_reg)
        with open(out_dir + 'class_reg.tex', 'w') as f:
            f.write(class_reg)
        with open(out_dir + 'importances.tex', 'w') as f:
            f.write(importances)
        if task_index > 0:
            with open(out_dir + 'scores_bal.tex', 'w') as f:
                f.write(scores_bal)
            with open(out_dir + 'class_bal.tex', 'w') as f:
                f.write(class_bal)
        # Save importances plot
        plt.savefig(out_dir + 'importances.png')
        plt.close()


def task_confusion_matrix(task_index, clf_row):
    filename_dict = {0: 'two.txt', 1: 'six.txt',
                     2: 'seven.txt', 3: 'seven.txt', 4: 'eight.txt'}
    filepath = 'templates/confusion/' + filename_dict[task_index]

    args_dict = dict()
    # Add Values
    confusion_matrix = clf_row.cnf_matrix
    class_names = pretty_class_labels(clf_row.class_labels)
    for i in range(confusion_matrix.shape[0]):
        class_key = 'name_{}'.format(i)
        args_dict[class_key] = class_names[i]
        for j in range(confusion_matrix.shape[1]):
            key = 'a{}_{}'.format(i, j)
            value = '{0:.2f}'.format(confusion_matrix[i][j])
            args_dict[key] = value

    args_dict['task_name'] = pretty_task_name(task_index)
    # Replace dict
    result = replace_template(filepath, args_dict)
    return result


def task_classifiers_score_latex(results_df, task_index, score_name, use_oversampled=False):
    assert score_name in ['fscore', 'precision', 'recall']

    # Filter results by oversampled
    results_df = results_df[results_df.oversample ==
                            use_oversampled].sort_values(by=['min_obs', 'num_features'])

    args_dict = dict()
    for _, row in results_df.iterrows():
        if row.scaler != 'StandardScaler':
            continue
        key = '{}_{}_{}'.format(row.model, row.min_obs, row.num_features)
        value = '{:.2f}'.format(row['test_{}'.format(score_name)] * 100)

        if key in args_dict:
            args_dict[key] = max(float(args_dict[key]), float(value))
        else:
            args_dict[key] = value

    args_dict['score_name'] = {
        'fscore': 'F1-Score', 'precision': 'Precision', 'recall': 'Recall'}[score_name]
    args_dict['task_name'] = pretty_task_name(task_index)
    args_dict['is_oversampled'] = 'balanced' if use_oversampled or task_index == 0 else 'unbalanced'
    # print(args_dict)

    result = replace_template('templates/classifiers.txt', args_dict)
    return result


def task_scores_latex(task_index, clf_row, use_oversampled=False):
    # Create Dict
    args_dict = dict()
    # print(clf_row)
    # Calculate Score and Support Keys
    for score_name in ['fscore', 'precision', 'recall', 'support']:
        current_score = clf_row['test_{}_by_class'.format(score_name)]
        for i, class_name in enumerate(pretty_class_labels(clf_row.class_labels)):
            key = '{}_{}'.format(
                class_name.lower().replace('-', ''), score_name)
            value = current_score[i]
            args_dict[key] = value if score_name == 'support' else '{0:.2f}'.format(
                value * 100)
            # print(key)
        # Calculate Total Value
        total_key = 'total_{}'.format(score_name)
        total_value = clf_row['test_{}'.format(
            score_name)] if score_name != 'support' else np.sum(clf_row['test_support_by_class'])
        args_dict[total_key] = total_value if score_name == 'support' else '{0:.2f}'.format(
            total_value * 100)

    # Add missing keys as None
    for missing_name in ['nontransient_precision', 'nontransient_recall',
                         'nontransient_fscore', 'nontransient_support', 'other_precision',
                         'other_recall', 'other_fscore', 'other_support']:
        if missing_name not in args_dict:
            args_dict[missing_name] = ''

    args_dict['task_name'] = pretty_task_name(task_index)
    args_dict['is_oversampled'] = 'Oversampled' if use_oversampled else 'Regular'

    filename = 'binary' if task_index == 0 else 'multi'

    # print(args_dict)

    result = replace_template(
        'templates/scores/{}.txt'.format(filename), args_dict)
    return result


def replace_template(file_path, args_dict):
    # open the file
    filein = open(file_path)
    # read it
    templ = Template(filein.read())
    # do the substitution
    # print(args_dict)
    result = templ.substitute(args_dict)
    return result


def _pretty_label_(label):
    label = label.replace('flux_percentile_ratio_mid', 'fpr')
    label = label.replace('percent_difference_flux_percentile', 'pdfp')
    label = label.replace('median_buffer_range_percentage', 'mbrp')
    label = label.replace('median_absolute_deviation', 'mad')
    label = label.replace('pair_slope_trend', 'pst')
    label = label.replace('percent_amplitude', 'p_amplitude')
    label = label.replace('amplitude', 'amp')
    label = label.replace('small_kurtosis', 'sk')
    label = label.replace('pst_last_30', 'pst_last30')
    label = label.replace('beyond1st', 'beyond1std')

    # label = label.replace('poly1_t1', 'poly1_1')
    #
    # label = label.replace('poly2_t1', 'poly2_1')
    # label = label.replace('poly2_t2', 'poly2_2')
    #
    # label = label.replace('poly3_t1', 'poly3_1')
    # label = label.replace('poly3_t2', 'poly3_2')
    # label = label.replace('poly3_t3', 'poly3_3')
    #
    # label = label.replace('poly4_t1', 'poly4_1')
    # label = label.replace('poly4_t2', 'poly4_2')
    # label = label.replace('poly4_t3', 'poly4_3')
    # label = label.replace('poly4_t4', 'poly4_4')
    return label


def importances_graph(clf, task_index, filepath):

    color_list = [
        mcolors.CSS4_COLORS["firebrick"],
        mcolors.CSS4_COLORS["lightseagreen"],
        mcolors.CSS4_COLORS["steelblue"],
        mcolors.CSS4_COLORS["yellowgreen"],
        mcolors.CSS4_COLORS["green"],
    ]

    task_name = pretty_task_name(task_index)
    title = task_name + ' Classification'

    importances = np.array([imp * 100 for imp in clf.feature_importances_])
    num_features = importances.shape[0]
    indices = np.argsort(importances)[::-1]
    label_names = [_pretty_label_(features.ALL_NO_CLASS[feature_index])
                   for feature_index in indices]
    # Setup Plot
    plt.rcParams["font.family"] = "Times New Roman"
    plt.figure(figsize=(20, 10))
    ax1 = plt.axis('tight')
    plt.gcf().subplots_adjust(bottom=0.20)
    plt.xlim((-.8, num_features))
    plt.title(title, fontsize=22, y=1.05)
    plt.xlabel('Features', fontsize=19)
    plt.ylabel('Feature importance (%)', fontsize=19)
    plt.bar(range(num_features), importances[indices], width=0.87,
            color=color_list[task_index], align="center", edgecolor='black')
    # color="r", yerr=std[indices], align="center")
    plt.xticks(range(num_features), label_names, rotation=90, fontsize=15)
    plt.yticks(np.arange(0, importances.max() + 1.5, 2), fontsize=15)

    # Setup Latex Graph
    args_dict = {
        'task_name': task_name,
        'filepath': filepath + task_name + '/' + 'importances.png',
    }
    result = replace_template('templates/importances.txt', args_dict)

    return result
    # plt.xlim([-1, importances.shape[0]])
    # plt.show()
    # return plt
