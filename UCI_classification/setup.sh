echo "Downloading data_py.zip from 'http://www.bioinf.jku.at/people/klambauer/data_py.zip'"
wget http://www.bioinf.jku.at/people/klambauer/data_py.zip
echo "Unzipping 'data_py.zip'"
unzip data_py.zip
cd data/
echo "Deleting a misplaced file"
rm abalone_py.dat
echo "Renaming certain directories for consistency: only letters a-z, numerals 0-9 and dashes '-'."
mv oocytes_merluccius_nucleus_4d/ oocytes-merluccius-nucleus-4d/
cd oocytes-merluccius-nucleus-4d/
mv oocytes_merluccius_nucleus_4d_py.dat oocytes-merluccius-nucleus-4d_py.dat 
cd ..
mv oocytes_merluccius_states_2f/ oocytes-merluccius-states-2f/
cd oocytes-merluccius-states-2f/
mv oocytes_merluccius_states_2f_py.dat oocytes-merluccius-states-2f_py.dat 
cd  ..
mv oocytes_trisopterus_nucleus_2f/ oocytes-trisopterus-nucleus-2f/
cd oocytes-trisopterus-nucleus-2f/
mv oocytes_trisopterus_nucleus_2f_py.dat oocytes-trisopterus-nucleus-2f_py.dat 
cd ..
mv oocytes_trisopterus_states_5b/ oocytes-trisopterus-states-5b/
cd oocytes-trisopterus-states-5b/
mv oocytes_trisopterus_states_5b_py.dat oocytes-trisopterus-states-5b_py.dat 
cd ..
cd ..
cp _datasets.py datasets.py
echo "ROOTDIR='$(pwd)'" >> datasets.py