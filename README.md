# financebot
Creating a chatbot for an ML coding challenge. This system will be able to interact w/ a user about their basic finance questions.

## Categories
* Check Balance (e.g., "How much do I have saved up in my account ending in
(x9898)?", "How much money do I have stashed up in my Bank of America?", "What is my
current BoA savings balance?", etc)

* Budgeting (e.g., "How much wiggle room do I have in my budget?", "How much
spare money do I have?", "How much money can I save each month?", etc)

* House affordability (e.g., "Can I afford a $2.3 million house?", "Can I buy a $2.3M
crib?", etc)

* N/A (e.g., "Can I buy a $3 million car?", "What day is it?")

## Usage

1. Clone the directory to your local computer
2. Run `pip3 install -r requirements.txt`
3. To simply use the chatbot, run `python3 src/chatbot.py`
    * Optional bool argument `--debug` (default False)
    * To exit, simply type `exit`, or press **Cmd-C**
4. To train the NER models, run `python3 src/train____NER.py`
5. To train the DNN model for categorization, run `python3 src/trainModel.py`
