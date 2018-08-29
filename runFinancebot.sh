echo "Welcome to Clerkie v0.1!"
sleep 3
echo "This is a short demo of the system I made for the Clerkie ML Challenge..."
sleep 3

echo "First I'll converse with the chatbot via text."
sleep 3
python3 src/chatbot.py

echo "This time, I'll converse via speech!"
sleep 3
python3 src/chatbot.py

echo "Finally, I will converse with the chatbot via text in debug mode, so you can see the classification results for the categorization and entity recognition."
sleep 3
python3 src/chatbot.py --debug True

