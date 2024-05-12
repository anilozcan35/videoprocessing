wget https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/r3yjf35hzr-1.zip
unzip r3yjf35hzr-1.zip -d ./
mv 'Shoplifting Dataset (2022) - CV Laboratory MNNIT Allahabad' shopliftingdata
unzip /content/shopliftingdata/Dataset.zip -d ./
cd /content/Dataset
pip install -U kora