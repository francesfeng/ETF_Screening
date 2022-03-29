mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"ffan1201@gmail.com\"\n\
" > ~/.streamlit/credentials.toml


echo "[theme]
primaryColor="#1830B7"
backgroundColor="#FFFFFF"
secondaryBackgroundColor="#F7F8FA"
textColor="#212121"
[runner]
magicEnabled = false
[global]
dataFrameSerialization = "legacy"
[server]
headless = true
port = $PORT
enableCORS = false
" > ~/.streamlit/config.toml
