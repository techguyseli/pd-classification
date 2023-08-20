def is_parkinsonian(participant_series):
    """
    Check if a participant is parkinsonian or not.

    Args:
        participantInfo (pandas.Series): A series representing the information about a participant.

    Returns:
        result (int): 1 if parkinsonian, 0 if healthy and -1 if has other dementia.
    """
    if participant_series.index.__contains__('Pathology') and isinstance(participant_series['Pathology'], str):
        if participant_series['Pathology'].lower() == 'none':
            return 0
        elif participant_series['Pathology'].lower() == 'parkinson':
            return 1
    else:
        if participant_series['Dementia'].lower() == 'no':
            return 0
        elif participant_series['Other Dementia'].lower() == 'parkinson':
            return 1
    return -1

