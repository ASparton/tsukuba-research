import numpy

def get_nb_classification_errors(model_predictions: numpy.ndarray, expected_output: numpy.ndarray) -> int:
    """Computes and returns the number of classification errors made by the model predictions."""
    
    nb_errors = 0
    for i in range(len(model_predictions)):
        if model_predictions[i] != expected_output[i]:
            nb_errors += 1
    return nb_errors

def get_nb_common_errors(predictions1: numpy.ndarray, predictions2: numpy.ndarray, expected_output: numpy.ndarray) -> int:
    nb_common_errors = 0
    for i in range(len(predictions1)):
        if ((predictions1[i] == predictions2[i]) and predictions1[i] != expected_output[i]):
            nb_common_errors += 1
    return nb_common_errors

def get_failure_probability(model_predictions: numpy.ndarray, expected_output: numpy.ndarray) -> float:
    return get_nb_classification_errors(model_predictions, expected_output) / len(expected_output)

def get_alpha(model1_predictions: numpy.ndarray, model2_predictions: numpy.ndarray, expected_output: numpy.ndarray) -> float:
    nb_model1_errors = get_nb_classification_errors(model1_predictions, expected_output)
    nb_model2_errors = get_nb_classification_errors(model2_predictions, expected_output)
    return get_nb_common_errors(model1_predictions, model2_predictions, expected_output) / min(nb_model1_errors, nb_model2_errors)

def get_beta(predictions1: numpy.ndarray, predictions2: numpy.ndarray, expected_output: numpy.ndarray) -> float:
    return (get_nb_common_errors(predictions1, predictions2, expected_output) / len(expected_output)) / get_failure_probability(predictions2, expected_output)

def get_tmsi_reliability(model1_predictions: numpy.ndarray, 
                         model2_predictions: numpy.ndarray, 
                         model3_predictions: numpy.ndarray, 
                         expected_output: numpy.ndarray) -> float:
    """Computes the reliability of a tmsi system"""
    
    return 1 - (
        get_alpha(model1_predictions, model2_predictions, expected_output) * get_failure_probability(model1_predictions, expected_output) +
        get_alpha(model1_predictions, model3_predictions, expected_output) * get_failure_probability(model1_predictions, expected_output) +
        get_alpha(model2_predictions, model3_predictions, expected_output) * get_failure_probability(model2_predictions, expected_output) -
        2 * get_alpha(model1_predictions, model2_predictions, expected_output) * 
        get_alpha(model1_predictions, model3_predictions, expected_output) * get_failure_probability(model1_predictions, expected_output)
    )
    
def get_smti_reliability(model1_predictions: numpy.ndarray, 
                         model2_predictions: numpy.ndarray, 
                         model3_predictions: numpy.ndarray, 
                         expected_output: numpy.ndarray) -> float:
    """Computes the reliability of a smti system"""
    
    return 1 - (
        get_beta(model2_predictions, model1_predictions, expected_output) * get_failure_probability(model1_predictions, expected_output) +
        get_beta(model3_predictions, model1_predictions, expected_output) * get_failure_probability(model1_predictions, expected_output) +
        get_beta(model3_predictions, model2_predictions, expected_output) * get_failure_probability(model2_predictions, expected_output) -
        2 * get_beta(model2_predictions, model1_predictions, expected_output) *
        get_beta(model3_predictions, model1_predictions, expected_output) *
        get_failure_probability(model1_predictions, expected_output)
    )
    
def get_dmdi_failure_probability(model1_predictions: numpy.ndarray, model2_predictions: numpy.ndarray, expected_output: numpy.ndarray):
    return (get_beta(model2_predictions, model1_predictions, expected_output) *
           get_alpha(model1_predictions, model2_predictions, expected_output) + 
           (1 - get_beta(model2_predictions, model1_predictions, expected_output)) *
           (get_failure_probability(model2_predictions, expected_output) - get_alpha(model1_predictions, model2_predictions, expected_output) *
            get_failure_probability(model1_predictions, expected_output)) /
           1 - get_failure_probability(model1_predictions, expected_output)) * get_failure_probability(model1_predictions, expected_output)
    
    
def get_tmti_reliability(model1_predictions: numpy.ndarray, 
                         model2_predictions: numpy.ndarray, 
                         model3_predictions: numpy.ndarray, 
                         expected_output: numpy.ndarray) -> float:
    """Computes the reliability of a tmti system"""
    
    return 1 - (
        get_dmdi_failure_probability(model1_predictions, model2_predictions, expected_output) +
        get_dmdi_failure_probability(model1_predictions, model3_predictions, expected_output) +
        get_dmdi_failure_probability(model2_predictions, model3_predictions, expected_output) -
        2 * get_dmdi_failure_probability(model1_predictions, model2_predictions, expected_output) *
        get_dmdi_failure_probability(model1_predictions, model3_predictions, expected_output) / 
        get_failure_probability(model1_predictions, expected_output)
    )