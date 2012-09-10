
#!/bin/sh
mkdir dist
mkdir dist/graphchi_v$1
cp -r src example_apps toolkits conf docs README.txt graphchi_xcode Makefile dist/graphchi_v$1
cd dist
#rm -rf  src/Eigen
find . -name '*.hg' -exec rm -r {} \;
find . -name '*.dSYM' -exec rm -r {} \;

tar -czf graphchi_src_v$1.tar.gz graphchi_v$1
