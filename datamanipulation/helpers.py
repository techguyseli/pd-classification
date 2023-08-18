def is_parkinsonian(participantInfo):
    """
    Check if a participant is parkinsonian or not.

    Args:
        participantInfo (pandas.Series): A series representing the information about a participant.

    Returns:
        result (int): 1 if parkinsonian, 0 if healthy and -1 if has other dementia.
    """
    if participantInfo.index.__contains__('Pathology') and isinstance(participantInfo['Pathology'], str):
        if participantInfo['Pathology'].lower() == 'none':
            return 0
        elif participantInfo['Pathology'].lower() == 'parkinson':
            return 1
    else:
        if participantInfo['Dementia'].lower() == 'no':
            return 0
        elif participantInfo['Other Dementia'].lower() == 'parkinson':
            return 1
    return -1

