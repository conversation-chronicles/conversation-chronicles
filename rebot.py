import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import colorful as cf

cf.use_true_colors()
cf.use_style('monokai')


class ReBot:
    def __init__(self):
        print(cf.bold | cf.purple("Preparing for multi-session chat with ReBot..."))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("Jihyoung/rebot-generation")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("Jihyoung/rebot-generation").to(self.device)
        self.summarizer = pipeline("summarization",
                                   model=AutoModelForSeq2SeqLM.from_pretrained("Jihyoung/rebot-summarization").to(
                                       self.device),
                                   tokenizer=AutoTokenizer.from_pretrained("Jihyoung/rebot-summarization"), device=0)
        self.single_session = []
        self.sequence = ""
        self.speaker = []

    def observe(self, observation):
        self.single_session.append(observation)

    def set_input(self):
        input_text = " ".join(self.single_session)
        input_text = "{} [now] {}".format(self.sequence, input_text)

        return input_text

    def generate(self, user_response):
        self.observe(user_response)

        input_text = self.set_input()

        inputs = self.tokenizer([input_text], return_tensors="pt").to(self.device)
        outputs = self.model.generate(inputs["input_ids"], max_new_tokens=128, temperature=1.0, top_p=.95,
                                      do_sample=True)
        rebot_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)

        if rebot_response != "[END]":
            self.observe(rebot_response)

        return rebot_response

    def summary(self):
        sequence = " ".join(self.single_session)
        event = self.summarizer(sequence)[0]['summary_text']

        return event

    def reset_history(self):
        self.single_session = []

    def run(self):
        def get_valid_input(prompt, default):
            while True:
                user_input = input(prompt)
                if user_input in ["Y", "N", "y", "n"]:
                    return user_input
                if user_input == "":
                    return default

        print(cf.orange("Which relationship should you and ReBot take?"))
        relationship = input(
            cf.orange(
                "Option: 1. Husband and Wife\t2. Parent and Child\t3. Co-workers\t"
                "4. Classmates\t5. Student and Teacher\t6. Patient and Doctor\t"
                "7. Employee and Boss\t8. Athlete and Coach\t9. Neighbors\t10. Mentee and Mentor: "))
        role = input(cf.orange("Specifically What is your role? "))

        if ' and ' in relationship:
            rel = relationship.strip().split(' and ')
            rel.remove(role)
            self.speaker = [role, rel[0]]
        else:
            self.speaker = [f'{relationship} A', f'{relationship} B']
        print(cf.blue(f"You are the {self.speaker[0]}!"))

        self.sequence = "<relationship> {}".format(relationship)

        session_count = 1
        event_history = []
        time_history = []
        while True:
            if session_count > 1:
                session_event = self.summary()
                print(cf.green(session_event))
                self.reset_history()
                event_history.append(session_event)
                print(cf.blue(f"Input the time interval between the previous session and the next session: "))
                time_interval = input(
                    cf.blue("Option: 1. A few hours after\t2. A few days after\t"
                            "3. A few weeks after\t4. A few months after\t5. A couple of years after: "))
                time_history.append(time_interval)
                self.sequence = "{} <{}> {}".format(self.sequence, time_interval, session_event)
            self.chat()
            session_count = session_count + 1
            continue_chat = get_valid_input(cf.purple("Start a new session with new event? [Y/N]: "), "Y")
            if continue_chat in ["N", "n"]:
                break

        print(cf.blue("Ending the chat with ReBot..."))

    def chat(self):
        print(cf.green(
            "Chat with ReBot! Input [NEXT] to switch to the next session and [END] to end the this session."))
        while True:
            user_input = input("You: ")
            if user_input == "[NEXT]" or user_input == "[END]":
                break
            user_input = "<{}> {} <{}>".format(self.speaker[0], user_input, self.speaker[1])
            response = self.generate(user_input)
            if response == "[END]":
                break
            print(cf.blue("ReBot: " + response))


def main():
    rebot = ReBot()
    rebot.run()


if __name__ == '__main__':
    main()
