mkdir -p ./data/TextClassification/agnews
mkdir -p ./data/TextClassification/dbpedia
mkdir -p ./data/TextClassification/yahoo_answers_topics

wget -P ./data/TextClassification/yahoo_answers_topics https://s3.amazonaws.com/fast-ai-nlp/yahoo_answers_csv.tgz
cd ./data/TextClassification/yahoo_answers_topics 
tar -zxvf yahoo_answers_csv.tgz
cd ../../..

cp classes/agnews/classes.txt ./data/TextClassification/agnews/classes.txt
cp classes/dbpedia/classes.txt ./data/TextClassification/dbpedia/classes.txt
cp classes/yahoo_answers_topics/classes.txt ./data/TextClassification/yahoo_answers_topics/classes.txt

python3 preprocess_data.py