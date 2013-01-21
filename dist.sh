
#!/bin/sh
mkdir dist
mkdir dist/graphchi_v$1

cd dist/graphchi_v$1
hg clone https://code.google.com/p/graphchi/ 
#rm -rf  src/Eigen
find . -name '*.hg' -exec rm -rf {} \;
find . -name '*.dSYM' -exec rm -fr {} \;
cd ..

tar -czf graphchi_src_v$1.tar.gz graphchi_v$1
