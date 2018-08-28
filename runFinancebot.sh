echo "Welcome to Clerkie v0.1!"
sleep 3
echo "This is a short demo of the system I made for the Clerkie ML Challenge..."
sleep 3
echo "First I'll converse with the chatbot via text in debug mode, so you can see the classification probabilities."
sleep 3
echo "Afterwards, I will restart, and this time converse via speech!"
sleep 3
python3 src/chatbot.py --debug True
python3 src/chatbot.py --debug False
