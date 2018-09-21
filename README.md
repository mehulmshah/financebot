# financebot
This is a financial advising chatbot. This system will be able to interact w/ a user about their basic finance questions.

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
3. To train the chatbot, run `python3 src/pretrainWV.py`. This will create 3 keras models based on a word2vec embedding layer pre-trained on Google News articles.
4. To deploy and use the actual chatbot, run `python3/chatbot.py`. Entering a request will retrieve responses from all 3 models. [WIP] Use combination of model outputs to make final decision, as each has strengths/weaknesses.

## Demo

Click [here](https://www.dropbox.com/s/pnc0p7mp4tj0sbe/D55701E8-4F0C-43BE-82B7-B893E489BD44_HQ.mp4?dl=0) to see a demo of the bot!
