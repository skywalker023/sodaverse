import re

def deduplicate_sequent_chars(string, chars):
    for c in chars:
        pattern = c + '{2,}'
        string = re.sub(pattern, c, string)

    return string

def capture_speakers(dialogue):
    """
    Capture any string between the newline and the colon (:)
    """
    # re_pattern = r'(?<=\n)[a-zA-Z\s]*(?=:)'
    re_pattern = r'(?<=\n)(.*)(?=:)'
    speakers = re.findall(re_pattern, "\n" + dialogue)
    speakers = [s.strip() for s in speakers]

    return speakers

def capture_utterances(dialogue):
    """
    Capture any string between colon (:) and the newline
    """
    dialogue = dialogue.replace(":\n", ": ")
    re_pattern = r'(?<=:)(.*)'
    utterances = re.findall(re_pattern, "\n" + dialogue)
    utterances = [u.strip() for u in utterances]

    return utterances

def split_speakers_and_utterances(dialogue):
    speakers = capture_speakers(dialogue)
    utterances = capture_utterances(dialogue)

    return speakers, utterances

def cleanup_dialogue(dialogue):
    dialogue = deduplicate_sequent_chars(dialogue, ["\n", "\t", " "])
    speakers, utterances = split_speakers_and_utterances(dialogue)
    clean_dialogue = ["{}: {}".format(s, u.strip()) for s, u in zip(speakers, utterances)]
    dialogue_string = "\n".join(clean_dialogue)

    output = {
        "dialogue": dialogue_string + "\n", # add newline for unifying the dialogue structure (=every utterance should end with a newline)
        "speakers": speakers,
        "utterances": utterances
    }

    return output