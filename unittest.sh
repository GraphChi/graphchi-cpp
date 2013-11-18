
# Following command makes the unittest.sh fail if any of the
# commands fail.
set -e

function display_name {
  echo "****************************************************************************"
  echo -n "*******************"
  echo -n $1
  echo "**********************"
  echo "****************************************************************************"
}


display_name "TESTING BASELINE"
  ./toolkits/collaborative_filtering/baseline --training=smallnetflix_mm --validation=smallnetflix_mm --minval=1 --maxval=5 --quiet=1  --algorithm=user_mean --clean_cache=1
display_name "TESTING ALS"
 ./toolkits/collaborative_filtering/als --training=smallnetflix_mm --validation=smallnetflix_mme --lambda=0.065 --minval=1 --maxval=5 --max_iter=6 --quiet=1 --clean_cache=1
display_name "TESTING ALS - RMSE VALIDATION STOP"
 ./toolkits/collaborative_filtering/als --training=smallnetflix_mm --validation=smallnetflix_mme --lambda=0.1 --minval=1 --maxval=5 --max_iter=100 --quiet=1 --halt_on_rmse_increase=3 --clean_cache=1
display_name "TESTING ALS SERIALIZATION"
 ./toolkits/collaborative_filtering/als --training=smallnetflix_mm --validation=smallnetflix_mme --lambda=0.065 --minval=1 --maxval=5 --max_iter=6 --quiet=1 --load_factors_from_file=1 --clean_cache=1

display_name "TESTING ALS - RATING"
./toolkits/collaborative_filtering/rating --algorithm=als --training=smallnetflix_mm  --quiet=1 --num_ratings=3
mv smallnetflix_mm.ids smallnetflix_mm.ids1
./toolkits/collaborative_filtering/rating --algorithm=als --training=smallnetflix_mm  --quiet=1 --num_ratings=3
diff smallnetflix_mm.ids smallnetflix_mm.ids1

display_name "TESTING ALS COORD"
 ./toolkits/collaborative_filtering/als_coord --training=smallnetflix_mm --validation=smallnetflix_mme --lambda=0.065 --minval=1 --maxval=5 --max_iter=6 --quiet=1 --clean_cache=1

display_name "TESTING SGD"
 ./toolkits/collaborative_filtering/sgd --training=smallnetflix_mm --validation=smallnetflix_mme --sgd_lambda=1e-4 --sgd_gamma=1e-4 --minval=1 --maxval=5 --max_iter=6 --quiet=1 --clean_cache=1

display_name "TESTING BIAS_SGD"
 ./toolkits/collaborative_filtering/biassgd --training=smallnetflix_mm --validation=smallnetflix_mme --biassgd_lambda=1e-4 --biassgd_gamma=1e-4 --minval=1 --maxval=5 --max_iter=6 --quiet=1 --clean_cache=1

display_name "TESTING BIAS_SGD SERIALIZATION"
 ./toolkits/collaborative_filtering/biassgd --training=smallnetflix_mm --validation=smallnetflix_mme --biassgd_lambda=1e-4 --biassgd_gamma=1e-4 --minval=1 --maxval=5 --max_iter=6 --quiet=1 --load_factors_from_file=1
display_name "TESTING SVD++"
 ./toolkits/collaborative_filtering/svdpp --training=smallnetflix_mm --validation=smallnetflix_mme --biassgd_lambda=1e-4 --biassgd_gamma=1e-4 --minval=1 --maxval=5 --max_iter=6 --quiet=1 --clean_cache=1

 ./toolkits/collaborative_filtering/rating2 --training=smallnetflix_mm --algorithm=svdpp --num_ratings=3 --quiet=1
display_name "TESTING SVD++ SERIALIZATION"
 ./toolkits/collaborative_filtering/svdpp --training=smallnetflix_mm --validation=smallnetflix_mme --biassgd_lambda=1e-4 --biassgd_gamma=1e-4 --minval=1 --maxval=5 --max_iter=6 --quiet=1 --load_factors_from_file=1 --clean_cache=1

display_name "TESTING NMF"
./toolkits/collaborative_filtering/nmf --training=reverse_netflix.mm --minval=1 --maxval=5 --max_iter=6 --quiet=1 --clean_cache=1

display_name "TESTING SVD"
./toolkits/collaborative_filtering/svd --training=smallnetflix_mm --nsv=3 --nv=5 --max_iter=5 --quiet=1 --tol=1e-1 --clean_cache=1 --validation=smallnetflix_mme --test=smallnetflix_mme

display_name "TESTING SVD-ONESIDED"
./toolkits/collaborative_filtering/svd_onesided --training=smallnetflix_mm --nsv=3 --nv=5 --max_iter=5 --quiet=1 --tol=1e-1 --clean_cache=1

display_name "TESTING RBM"
./toolkits/collaborative_filtering/rbm --training=smallnetflix_mm --validation=smallnetflix_mme --minval=1 --maxval=5 --max_iter=6 --quiet=1 --clean_cache=1

display_name "TESTING WALS"  
./toolkits/collaborative_filtering/wals --training=time_smallnetflix --validation=time_smallnetflixe --lambda=0.065 --minval=1 --maxval=5 --max_iter=6 --K=27 --quiet=1 --clean_cache=1

display_name "TESTING WALS - RATING"
./toolkits/collaborative_filtering/rating --training=time_smallnetflix --algorithm=wals --quiet=1 --num_ratings=3
display_name "TESTING ALS-TENSOR"  
./toolkits/collaborative_filtering/als_tensor --training=time_smallnetflix --validation=time_smallnetflixe --lambda=0.065 --minval=1 --maxval=5 --max_iter=6 --K=27 --quiet=1 --clean_cache=1
display_name "TESTING TIME-SVD++"
./toolkits/collaborative_filtering/timesvdpp --training=time_smallnetflix --validation=time_smallnetflixe --minval=1 --maxval=5 --max_iter=6 --quiet=1 --clean_cache=1
display_name "TESTING LIBFM"
./toolkits/collaborative_filtering/libfm --training=time_smallnetflix --validation=time_smallnetflixe --minval=1 --maxval=5 --max_iter=6 --quiet=1 --clean_cache=1
display_name "TESTING BIAS_SGD2 - LOGISTIC LOSS"
./toolkits/collaborative_filtering/biassgd2 --training=smallnetflix_mm --minval=1 --maxval=5 --validation=smallnetflix_mme --biassgd_gamma=1e-2 --biassgd_lambda=1e-2 --max_iter=6 --quiet=1 --loss=logistic --biassgd_step_dec=0.99999 --clean_cache=1
display_name "TESTING BIAS_SGD2 - ABS LOSS"
./toolkits/collaborative_filtering/biassgd2 --training=smallnetflix_mm --minval=1 --maxval=5 --validation=smallnetflix_mme --biassgd_gamma=1e-2 --biassgd_lambda=1e-2 --max_iter=6 --quiet=1 --loss=abs --biassgd_step_dec=0.99999 --clean_cache=1
display_name "TESTING BIAS_SGD2 - SQUARE LOSS"
./toolkits/collaborative_filtering/biassgd2 --training=smallnetflix_mm --minval=1 --maxval=5 --validation=smallnetflix_mme --biassgd_gamma=1e-2 --biassgd_lambda=1e-2 --max_iter=6 --quiet=1 --loss=square --biassgd_step_dec=0.99999 --clean_cache=1
#display_name "TESTING PMF"
# ./toolkits/collaborative_filtering/pmf --training=smallnetflix_mm --validation=smallnetflix_mme --quiet=1 --minval=1 --maxval=5 --max_iter=10 --pmf_burn_in=5 --test=smallnetflix_mme --regnormal=0 --clean_cache=1
display_name "TESTING ITEMCF"
./toolkits/collaborative_filtering/itemcf --training=smallnetflix_mm --nshards=1 --quiet=1 --K=10 --clean_cache=1
display_name "TESTING ITEMCF - AIOLLI ASYM COST"
./toolkits/collaborative_filtering/itemcf --training=smallnetflix_mm --nshards=1 --quiet=1 --distance=3 --K=10 --clean_cache=1
display_name "ITEM-SIM-TO-RATING"
rm -fR ./toolkits/collaborative_filtering/unittest/itemsim2rating.unittest.graph.*
./toolkits/collaborative_filtering/itemsim2rating --training=./toolkits/collaborative_filtering/unittest/itemsim2rating.unittest.graph --similarity=./toolkits/collaborative_filtering/unittest/itemsim2rating.unittest.similarity  --K=4 execthreads 1 --nshards=1 --quiet=0 --undirected=1 --debug=1
diff ./toolkits/collaborative_filtering/unittest/itemsim2rating.unittest.graph-rec ./toolkits/collaborative_filtering/unittest/itemsim2rating.unittest
display_name "TESTING ITEMCF - CORRECTNESS"
rm -fR ./toolkits/collaborative_filtering/unittest/itemcf.unittest.graph.*
./toolkits/collaborative_filtering/itemcf --training=./toolkits/collaborative_filtering/unittest/itemcf.unittest.graph --min_allowed_intersection=2 --K=5 --nshards=1 --quiet=1 execthreads 1
sh ./toolkits/collaborative_filtering/topk.sh ./toolkits/collaborative_filtering/unittest/itemcf.unittest.graph
#diff ./toolkits/collaborative_filtering/unittest/itemcf.unittest.graph-topk ./toolkits/collaborative_filtering/unittest/itemcf.unittest.graph-topk-correct
a=`grep "0.400000" ./toolkits/collaborative_filtering/unittest/itemcf.unittest.graph-topk | wc -l`
if [ $a -ne 3 ]; then
  echo "Failed unittest!"
  exit 1
fi
display_name "MAP METRIC - test 1"
./toolkits/collaborative_filtering/metric_eval --training=./toolkits/collaborative_filtering/unittest/metric_eval.unittest4 --test=./toolkits/collaborative_filtering/unittest/metric_eval.unittest3 --K=3 
display_name "MAP METRIC - test 2"
./toolkits/collaborative_filtering/metric_eval --training=./toolkits/collaborative_filtering/unittest/metric_eval.unittest2 --test=./toolkits/collaborative_filtering/unittest/metric_eval.unittest2 --K=3 
display_name "TOP K"
./toolkits/parsers/topk --training=./toolkits/collaborative_filtering/unittest/topk.unittest --K=3 --quiet=1
diff ./toolkits/collaborative_filtering/unittest/topk.unittest.ids ./toolkits/collaborative_filtering/unittest/topk.unittest.ids.correct
display_name "ITEMCF3"
./toolkits/collaborative_filtering/itemcf3 --training=./toolkits/collaborative_filtering/unittest/itemcf3.unittest.graph --distance=9 --debug=0 --quiet=1 --execthreads=1 
#diff ./toolkits/collaborative_filtering/unittest/itemcf3.unittest.correct ./toolkits/collaborative_filtering/unittest/itemcf3.unittest.graph.out0
a=`grep "2 1 0.6666" ./toolkits/collaborative_filtering/unittest/itemcf3.unittest.graph.out0 | wc -l`
if [ $a -ne 1 ]; then
  echo "Failed unittest!"
  exit 1
fi
b=`grep "3 1 0.3333" ./toolkits/collaborative_filtering/unittest/itemcf3.unittest.graph.out0 | wc -l`
if [ $b -ne 1 ]; then
  echo "Failed unittest!"
  exit 1
fi
display_name "GENSGD"
./toolkits/collaborative_filtering/gensgd --training=smallnetflix_mm --validation=smallnetflix_mme --from_pos=0 --to_pos=1 --val_pos=2 --rehash=1 --has_header_titles=0 --debug=0 --file_columns=3 --quiet=1 --max_iter=3 --train_only=1
./toolkits/collaborative_filtering/gensgd --training=smallnetflix_mm --validation=smallnetflix_mme --from_pos=0 --to_pos=1 --val_pos=2 --rehash=1 --has_header_titles=0 --debug=0 --file_columns=3 --quiet=1 --max_iter=3 --validation_only=1 
display_name "K-FOLD cross validation"
./toolkits/collaborative_filtering/als --training=smallnetflix_mm --validation=smallnetflix_mme --lambda=0.065 --minval=1 --maxval=5 --max_iter=6 --quiet=1 --kfold_cross_validation=10 --kfold_cross_validation_index=3 
./toolkits/collaborative_filtering/als --training=smallnetflix_mm --validation=smallnetflix_mme --lambda=0.065 --minval=1 --maxval=5 --max_iter=6 --quiet=1 --kfold_cross_validation=10 --kfold_cross_validation_index=4 

display_name "CLiMF"
./toolkits/collaborative_filtering/climf --training=smallnetflix_mm --validation=smallnetflix_mme --binary_relevance_thresh=4 --sgd_gamma=1e-6 --max_iter=6 --quiet=1 --sgd_step_dec=0.9999 --sgd_lambda=1e-6 --clean_cache=1
