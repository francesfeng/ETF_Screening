mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"ffan1201@gmail.com\"\n\
" > ~/.streamlit/credentials.toml


echo "\
[theme]\n\
primaryColor=\"#1830B7\"\n\
backgroundColor=\"#FFFFFF\"\n\
secondaryBackgroundColor=\"#F7F8FA\"\n\
textColor=\"#212121\"\n\
[runner]\n\
magicEnabled = false\n\
[client]\n\
showErrorDetails = false\n\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
" > ~/.streamlit/config.toml
