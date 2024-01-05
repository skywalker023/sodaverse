import os
import json
import argparse
from pathlib import Path

import openai
import numpy as np
from tqdm import tqdm
import colorful as cf
cf.use_true_colors()
cf.use_style('monokai')

from utils.dialogue_utils import cleanup_dialogue
import tasks.dataset_loaders as loader
from agents.gpt import GPT3BaseAgent, ChatGPTBaseAgent

PROJECT_HOME = Path(__file__).parent.resolve()
DATA_DIR = 'data'
DATA_DIR_PATH = os.path.join(PROJECT_HOME, DATA_DIR)

class CO3():
    def __init__(self, args):
        self.args = args
        self.args.dump_dir = self.args.run_id + ":{}_out_of_{}".format(args.split_num, args.split_data)
        self.atomic10x = self.load_atomic10x()

        self.set_llm_and_instruction(args)
        self.build_output_file(args) # if the directory already exists, it loads the existing args from the directory
        self.print_args()
        self.print_soda()

    def set_llm_and_instruction(self, args):
        if args.model.startswith('text-davinci-'):
            self.llm = GPT3BaseAgent(args.__dict__)
            self.narrative_prompt = "Rewrite this story with more specific details in two or three sentences:"
            self.dialogue_prompt = "The following is a long in-depth conversation happening in the scene between person 1 and person 2 with multiple turns."
        elif args.model.startswith('gpt-'):
            self.llm = ChatGPTBaseAgent(args.__dict__)
            self.narrative_prompt = "Rewrite this story with more specific details in two or three sentences:"
            self.dialogue_prompt = "Generate an in-depth conversation happening in the scene between person 1 and person 2 with multiple turns."
        else:
            # TODO: add other LLMs here!
            raise NotImplementedError

        self.prompt = [self.narrative_prompt, self.dialogue_prompt]
        self.prompt_suffix = "\nPerson 1:"
        self.prompt_suffix2 = "\nPerson 2:"

    def identify_interlocutor_with_gpt3(self, prompt):
        response = openai.Completion.create(
            model="gpt-3.5-turbo-instruct",
            prompt=prompt,
            temperature=0,
            max_tokens=16,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=["."],
            echo=True # XXX: to get the full output
        )

        return response['choices'][0]['text'].strip()

    def set_prompt_for_dialogue(self, text, **speakers):
        """
        Set prompt for dialogue generation with the interlocutors.
        """
        speaker_prefix = "\n" + speakers['x'] + ":"
        command_prompt = self.dialogue_prompt.replace("person 1", speakers['x'])

        # if there's PersonX and PersonY in the narrative, use them as the speakers.
        if 'y' in speakers.keys() and speakers['y'] != "":
            command_prompt = command_prompt.replace("person 2", speakers['y'])
            if 'z' in speakers.keys() and speakers['z'] != "":
                command_prompt = command_prompt.replace("with", "and " + speakers['z'] + " with")
            prompt = text + " " + command_prompt + speaker_prefix
        else: # if there's only PersonX in the narrative, prompt GPT-3 to figure out who is the most natural interlocutor.
            trimmed_prompt = command_prompt.split("person 2")[0].strip()
            prompt_to_complete = text + " " + trimmed_prompt
            command_prompt = self.identify_interlocutor_with_gpt3(prompt_to_complete)
            if not command_prompt.endswith("."):
                command_prompt = command_prompt + "."
            prompt = command_prompt + speaker_prefix

        return prompt

    def print_args(self):
        # sorted_args = sorted(self.args.__dict__.items())
        print("\n=======================================")
        for idx, (k, v) in enumerate(self.args.__dict__.items()):
            if idx != 0:
                print("---------------------------------------")
            print(k, " : ", v)
        print("=======================================\n")

    def print_soda(self):
        print()
        with open(os.path.join(PROJECT_HOME, 'assets', 'soda.txt'), 'r') as fp:
            for idx, line in enumerate(fp.readlines()):
                if idx in [0, 2, 4, 6]:
                    print(cf.bold | cf.ghostWhite(line), end="")
                elif idx in [1, 3, 5]:
                    print(cf.bold | cf.magenta(line), end="")
                else:
                    print(cf.bold | cf.blue(line), end="")
        print()
        print(cf.bold | cf.ghostWhite("[[ SODA coming up right now! ]]"))
        print()

    def run(self):
        last_save_point = self._load_last_save_point()

        t = tqdm(total=len(self.atomic10x))
        for current_idx, data_input in self.atomic10x.iterrows():
            if current_idx <= last_save_point:
                t.update(1)
                continue

            if self.args.generation_limit is not None:
                if current_idx > self.args.generation_limit:
                    break

            sentence_form_triple = data_input['input_text']
            narrative_result = self._collect_narrative(sentence_form_triple, **data_input)
            output = self._collect_dialogue(narrative_result['narrative'], **data_input)
            output['narrative_prompt'] = narrative_result['narrative_prompt']
            output['narrative'] = narrative_result['narrative']

            if current_idx % self.args.display_frequency == 0:
                print()
                print(cf.bold | cf.yellow("[ Triple ] " + data_input['head'] + " || " + data_input['relation'] + " || " + data_input['tail']))
                print(cf.bold | cf.lightOrange("[ Sentence-form ] " + data_input['input_text']))
                print(cf.bold | cf.green("[ Narrative ] " + output['narrative']))
                first_speaker = output['dialogue_prompt'].split("\n")[-1]
                print(cf.bold | cf.blue("[ Dialogue ]"))
                print(cf.blue(first_speaker + output['cumulative_dialogue']))
                print()

            self._dump_output(current_idx, output, **data_input)
            t.update(1)

    def _generate_narrative(self, text):
        prompt = text + " " + self.narrative_prompt

        narrative = self.llm.interact(prompt)
        narrative = narrative.replace("\n\n", "\n").replace("\n", " ").strip()
        result = {
            'narrative': narrative,
            'narrative_prompt': prompt
        }

        return result

    def _generate_dialogue(self, text, **data_input):
        """
        Generate dialogue with the given narrative text.
        """
        speakers = {'x': data_input['x'], 'y': data_input['y'], 'z': data_input['z']}
        _prompt = prompt = self.set_prompt_for_dialogue(text, **speakers)

        raw_dialogue = self.llm.interact(prompt)
        result = self._parse_dialogue_output(raw_dialogue, prompt, **data_input)
        length = result['num_responses']

        # if it contained "\n\n" in the first place, maybe that caused the dialogue to stop. So, continue generating with the cleaned dialogue
        if "\n\n" in raw_dialogue or length < self.args.min_dialogue_turn:
            continue_generation = True
        else:
            continue_generation = False

        # Try continuing the generation after we clean up the dialogue format in self._parse_output()
        continuation_count = self.args.conversation_continuation_count
        while continue_generation:
            # print(cf.bold | cf.yellow("Continuing the dialogue..."))
            prompt += result['dialogue']
            raw_dialogue = self.llm.interact(prompt)
            result = self._parse_dialogue_output(raw_dialogue, prompt, previous_result=result, **data_input)
            continuation_count -= 1
            length += result['num_responses']
            continue_generation = result['continue_generation']

            # if it has several utterances and the continuation_count is not left, stop.
            if continuation_count == 0:
                # print(cf.bold("Stopping the dialogue because it ran out of counts!"))
                continue_generation = False

        result['dialogue_prompt'] = _prompt

        return result

    def _collect_narrative(self, text, **data_input):
        attempt_count = self.args.generation_attempt_count

        narrative = None
        generated_narratives = []
        while narrative is None:
            result = self._generate_narrative(text)
            narrative = result['narrative']
            generated_narratives.append(narrative)
            result['suspended'] = False # default flag

            narrative_sentences = narrative.split(". ")
            if len(narrative_sentences) >= 4:
                attempt_count -= 1
                print(cf.bold | cf.purple("Too long in length! Attempt count left: " + str(attempt_count)))
                narrative = None
            elif narrative == text:
                print(cf.bold | cf.purple("The generated narrative is the same as the literal!"))
                narrative = None
                del generated_narratives[-1]
            elif len(narrative_sentences) != len(set(narrative_sentences)):
                print(cf.bold | cf.purple("Repetitive sentences in the narrative!"))
                narrative = None

            if attempt_count == 0:
                print(cf.bold | cf.magenta("Tried enough!"))
                result['suspended'] = True
                break

        if narrative is None:
            # choose from the existing ones
            print(cf.bold("Choosing the shortest one among the generated ones!"))
            sorted_narratives = sorted(generated_narratives, key=len)
            narrative = sorted_narratives[0]

        result['narrative'] = narrative
        result['all_generated_narratives'] = generated_narratives

        return result

    def _collect_dialogue(self, text, **data_input):
        attempt_count = self.args.generation_attempt_count
        repetition_tolerance = self.args.repetition_tolerance

        cumulative_dialogue = None
        generated_dialogues = []
        while cumulative_dialogue is None:
            result = self._generate_dialogue(text, **data_input)
            cumulative_dialogue = result['cumulative_dialogue']

            unique_utterances = set(result['cumulative_utterances'])
            n_repetitive_utterances = len(result['cumulative_utterances']) - len(unique_utterances)
            result['repetition'] = False # default flag
            result['suspended'] = False # default flag

            generated_dialogues.append(cumulative_dialogue)

            if len(result['cumulative_utterances']) < self.args.min_dialogue_turn:
                cumulative_dialogue = None
                attempt_count -= 1
                print(cf.bold | cf.purple("The dialogue is too short! Attempt count left: " + str(attempt_count)))
            elif len(result['cumulative_speakers']) < 2:
                cumulative_dialogue = None
                attempt_count -= 1
                print(cf.bold | cf.purple("There are less than two speakers! Attempt count left: " + str(attempt_count)))
            elif n_repetitive_utterances > 0:
                repetition_tolerance -= 1
                print(cf.bold | cf.purple("Has " + str(n_repetitive_utterances) + " repetitive utterances! Generating the dialogue again..."))
                print(cf.bold | cf.purple("Repetition tolerance:", repetition_tolerance))
                print(cf.bold | cf.yellow(result['dialogue_prompt']))
                print(cumulative_dialogue)
                if repetition_tolerance == 0:
                    result['repetition'] = True
                else:
                    cumulative_dialogue = None
                    del generated_dialogues[-1]

            if attempt_count == 0:
                print(cf.bold | cf.magenta("Tried enough!"))
                result['suspended'] = True
                break

        if cumulative_dialogue is None:
            # choose from the existing ones
            sorted_dialogues = sorted(generated_dialogues, key=len)
            cumulative_dialogue = sorted_dialogues[-1]
            print(cf.bold("Choosing the longest one among the generated ones!"))

        result['all_generated_dialogues'] = generated_dialogues

        return result

    def _parse_dialogue_output(self, raw_dialogue, prompt, previous_result=None, **data_input):
        # need to add the first speaker prefix
        if previous_result is None:
            starting_speaker = prompt.split()[-1]
            raw_dialogue = starting_speaker + raw_dialogue
        else:
            starting_speaker = previous_result['speakers'][0]

        # clean up dialogue
        clean_dialogue = cleanup_dialogue(raw_dialogue)
        dialogue = clean_dialogue['dialogue']
        num_responses = len(clean_dialogue['speakers'])

        # if it's a newly generated dialogue
        continue_generation = True
        if previous_result is None:
            cumulative_dialogue = dialogue
            cumulative_speakers = clean_dialogue['speakers']
            cumulative_utterances = clean_dialogue['utterances']
        # if we are continuing the dialogue, cumulate the dialogue
        else:
            cumulative_dialogue = previous_result['cumulative_dialogue']
            cumulative_utterances = previous_result['cumulative_utterances']
            cumulative_speakers = previous_result['cumulative_speakers']

            if dialogue == "\n":
                # if the current output is empty make sure to stop
                print(cf.bold("Stopping the dialogue because nothing was generated"))
                continue_generation = False
            elif num_responses == 1:
                # if GPT-3 only adds a single utterance, maybe it has nothing more to say!
                print(cf.bold("Stopping the dialogue because it has probably nothing more to say!"))
                continue_generation = False
            else:
                cumulative_dialogue = cumulative_dialogue + dialogue
                cumulative_utterances = cumulative_utterances + clean_dialogue['utterances']
                cumulative_speakers = cumulative_speakers + clean_dialogue['speakers']

        result = {
            'dialogue': dialogue,
            'speakers': clean_dialogue['speakers'],
            'utterances': clean_dialogue['utterances'],
            'num_responses': num_responses,
            'cumulative_dialogue': cumulative_dialogue.removeprefix(starting_speaker), # remove the first speaker prefix for continuing the dialogue because it's already in the prompt.
            'cumulative_speakers': cumulative_speakers,
            'cumulative_utterances': cumulative_utterances,
            'continue_generation': continue_generation,
        }

        return result

    def load_atomic10x(self):
        _df = loader.load('atomic10x')
        whole_df = _df[_df['x_relations']].copy().reset_index() # XXX: because we are only using a subset, so there will be some missing indices
        whole_df.rename(columns={'index': 'original_index', 'named_literal': 'input_text'}, inplace=True)
        df_chunks = np.array_split(whole_df, self.args.split_data)
        df = df_chunks[self.args.split_num]

        return df

    def _load_last_save_point(self):
        if os.path.exists(self.last_save_point_file):
            with open(self.last_save_point_file, 'r') as fp:
                last_save_point = int(fp.readlines()[0].strip())
        else:
            last_save_point = -1

        return last_save_point

    def build_output_file(self, args):
        """
            This function builds the output directory for dumping the results.
            If the directory already exists,
            it will automatically pick up where it stopped before and load the existing hyper parameters.
        """
        assert args.dump_dir is not None

        self.output_dump_location = os.path.join(DATA_DIR_PATH, "soda:" + args.dump_dir)
        os.makedirs(self.output_dump_location, exist_ok=True)
        args_file = os.path.join(self.output_dump_location, 'args.json')

        # if previously used args already exist, load them and override
        if os.path.exists(args_file):
            with open(args_file, 'r') as fp:
                previous_args = json.load(fp)
            for k, v in previous_args.items():
                setattr(self.args, k, v)
            self.prompt = previous_args['prompt']
        else:
            # save the arguments inside the dumping directory
            args_dict = vars(args).copy()
            del args_dict['generation_attempt_count']
            del args_dict['generation_limit']
            del args_dict['display_frequency']
            args_dict['prompt'] = self.prompt
            with open(args_file, 'w') as fp:
                json.dump(args_dict, fp)

        self.dump_file = os.path.join(self.output_dump_location, 'dump_file_' + args.dataset + '.jsonl')
        self.last_save_point_file = os.path.join(self.output_dump_location, 'last_save_point_' + args.dataset + '.txt')

    def _dump_output(self, idx, output, **data_input):
        file_name = self.dump_file
        # update save point
        with open(self.last_save_point_file, 'w') as fp:
            fp.write(str(idx))

        with open(file_name, 'a') as fp:
            del output['dialogue']
            del output['speakers']
            del output['utterances']
            del output['num_responses']

            data = {'index': int(idx), **output, **data_input}
            fp.write(json.dumps(data) + '\n')

def main(args):
    soda_maker = CO3(args)
    soda_maker.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arguments for generating dialogues using instruct gpt3')
    parser.add_argument('--dataset',
                        type=str,
                        default='atomic10x')
    parser.add_argument('--run-id',
                        type=str,
                        default='vanilla',
                        help='the name of the directory where the output will be dumped')
    parser.add_argument('--generation-limit',
                        type=int,
                        default=None,
                        help='the number of dialogues that this run will generate. If None, it will generate with the entire given dataset.')
    parser.add_argument('--display-frequency',
                        type=int,
                        default=1,
                        help='the frequency of displaying the generation results')
    parser.add_argument('--model',
                        type=str,
                        default='gpt-3.5-turbo-1106',
                        help='which LLM to use')
    parser.add_argument('--temperature',
                        type=float,
                        default=0.9,
                        help="control randomness: lowering results in less random completion")
    parser.add_argument('--top-p',
                        type=float,
                        default=0.95,
                        help="nucleus sampling")
    parser.add_argument('--frequency-penalty',
                        type=float,
                        default=1.0,
                        help="decreases the model's likelihood to repeat the same line verbatim")
    parser.add_argument('--presence-penalty',
                        type=float,
                        default=0.6,
                        help="increases the model's likelihood to talk about new topics")
    parser.add_argument('--max-tokens',
                        type=int,
                        default=1024,
                        help='maximum number of tokens to generate')
    parser.add_argument('--min-dialogue-turn',
                        type=int,
                        default=6,
                        help='minimum number of turns for a dialogue (if gpt-3 still fails to generate longer than after generation-attempt-count, it will let the dialogue be)')
    parser.add_argument('--conversation-continuation-count',
                        type=int,
                        default=1,
                        help='maximum number of attempts to continue the current conversation')
    parser.add_argument('--generation-attempt-count',
                        type=int,
                        default=2,
                        help='maximum number of attempts to generate a dialogue again')
    parser.add_argument('--repetition-tolerance',
                        type=int,
                        default=1,
                        help='maximum number of generation attempts when repetitive utterance is present in the dialogue')
    parser.add_argument('--split-data',
                        type=int,
                        default=15,
                        help='how many splits for the data?')
    parser.add_argument('--split-num',
                        type=int,
                        default=0,
                        help='access which data split?')
    args = parser.parse_args()
    main(args)
