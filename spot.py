

def load_dataset():
    # http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html
    # load the dataset, returns X and y elements
    return make_classification(n_samples=1000, n_classes=2, random_state=1)

def load_dataset():
    """
    Loads the given dataset.
    Needs to be configured before running the program

    :return: X, y
    """
    X, y = None, None
    return X, y

def define_models():
    """
    Defines and intantiates the models we want to
    evaluate

    {name : pipeline-object}
    :return: dict - scikit learn pipelines
    """
    models = {}
    return models

def make_pipeline(model):
    """
    Feature preparation for a model

    :return: model pipelined
    """
    steps = list()

    # TODO: why standardize and normalize?
    steps.append(('standardize', StandardScaler()))
    steps.append(('normalize', MinMaxScaler()))


    steps.append(('model', model))

    pipeline = Pipeline(steps=steps)
    return pipeline


def evaluate_models(X, y, models, folds=10, metric='accuracy'):
    # evaluate a dict of models {name:object}, returns {name:score}
    results = dict()
    for name, model in models.items():
            # evaluate the model
            scores = robust_evaluate_model(X, y, model, folds, metric)
            # show process
            if scores is not None:
                    # store a result
                    results[name] = scores
                    mean_score, std_score = mean(scores), std(scores)
                    print('>%s: %.3f (+/-%.3f)' % (name, mean_score, std_score))
            else:
                    print('>%s: error' % name)
    return results

def robust_evaluate_model(X, y, model, folds, metric):
    # evaluate a model and try to trap errors and and hide warnings
    scores = None
    try:
            with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    scores = evaluate_model(X, y, model, folds, metric)
    except:
            scores = None
    return scores


def summarize_results(results, maximize=True, top_n=10):
    # print and plot the top n results
    # check for no results
    if len(results) == 0:
            print('no results')
            return
    # determine how many results to summarize
    n = min(top_n, len(results))
    # create a list of (name, mean(scores)) tuples
    mean_scores = [(k,mean(v)) for k,v in results.items()]
    # sort tuples by mean score
    mean_scores = sorted(mean_scores, key=lambda x: x[1])
    # reverse for descending order (e.g. for accuracy)
    if maximize:
            mean_scores = list(reversed(mean_scores))
    # retrieve the top n for summarization
    names = [x[0] for x in mean_scores[:n]]
    scores = [results[x[0]] for x in mean_scores[:n]]
    # print the top n
    print()
    for i in range(n):
            name = names[i]
            mean_score, std_score = mean(results[name]), std(results[name])
            print('Rank=%d, Name=%s, Score=%.3f (+/- %.3f)' % (i+1, name, mean_score, std_score))
    # boxplot for the top n
    pyplot.boxplot(scores, labels=names)
    _, labels = pyplot.xticks()
    pyplot.setp(labels, rotation=90)
    pyplot.savefig('spotcheck.png')

if __name__ == '__main__':
    X, y = load_dataset()
    models = define_models()



