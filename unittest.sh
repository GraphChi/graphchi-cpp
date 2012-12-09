echo "TESTING BASELINE"
  ./toolkits/collaborative_filtering/baseline --training=smallnetflix_mm --validation=smallnetflix_mm --minval=1 --maxval=5 --quiet=1  --algorithm=user_mean
echo "TESTING ALS"
 ./toolkits/collaborative_filtering/als --training=smallnetflix_mm --validation=smallnetflix_mme --lambda=0.065 --minval=1 --maxval=5 --max_iter=6 --quiet=1
echo "TESTING SGD"
 ./toolkits/collaborative_filtering/sgd --training=smallnetflix_mm --validation=smallnetflix_mme --sgd_lambda=1e-4 --sgd_gamma=1e-4 --minval=1 --maxval=5 --max_iter=6 --quiet=1
echo "TESTING BIAS_SGD"
 ./toolkits/collaborative_filtering/biassgd --training=smallnetflix_mm --validation=smallnetflix_mme --biassgd_lambda=1e-4 --biassgd_gamma=1e-4 --minval=1 --maxval=5 --max_iter=6 --quiet=1
echo "TESTING SVD++"
 ./toolkits/collaborative_filtering/svdpp --training=smallnetflix_mm --validation=smallnetflix_mme --biassgd_lambda=1e-4 --biassgd_gamma=1e-4 --minval=1 --maxval=5 --max_iter=6 --quiet=1
echo "TESTING NMF"
./toolkits/collaborative_filtering/nmf --training=reverse_netflix.mm --minval=1 --maxval=5 --max_iter=20 --quiet=1
echo "TESTING SVD"
./toolkits/collaborative_filtering/svd --training=smallnetflix_mm --nsv=3 --nv=10 --max_iter=5 --quiet=1 --tol=1e-1
echo "TESTING SVD-ONESIDED"
./toolkits/collaborative_filtering/svd_onesided --training=smallnetflix_mm --nsv=3 --nv=10 --max_iter=5 --quiet=1 --tol=1e-1
echo "TESTING RBM"
./toolkits/collaborative_filtering/rbm --training=smallnetflix_mm --validation=smallnetflix_mme --minval=1 --maxval=5 --max_iter=6 --quiet=1
echo "TESTING ALS-TENSOR"  
./toolkits/collaborative_filtering/als_tensor --training=time_smallnetflix --validation=time_smallnetflixe --lambda=0.065 --minval=1 --maxval=5 --max_iter=6 --K=27 --quiet=1
echo "TESTING TIME-SVD++"
./toolkits/collaborative_filtering/timesvdpp --training=time_smallnetflix --validation=time_smallnetflixe --minval=1 --maxval=5 --max_iter=6 --quiet=1
echo "TESTING LIBFM"
./toolkits/collaborative_filtering/libfm --training=time_smallnetflix --validation=time_smallnetflixe --minval=1 --maxval=5 --max_iter=6 --quiet=1
