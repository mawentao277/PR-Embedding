#$1 input file, two columns: post"\t"reply;
#$2 output file, four columns: post"\t"reply"\t"p_align"\t"r_align;
set -x
echo "Begin to calculate the word alignment dict..."
align_dir=./data/align_data
if [ ! -d ${align_dir} ]; then
  mkdir -p ${align_dir}
fi
if [ ! -d ./data ]; then
  mkdir -p ./data
fi
cut -f1 $1 > ${align_dir}/post
cut -f2 $1 > ${align_dir}/reply
./word-alignment/GIZA++-v2/plain2snt.out ${align_dir}/post ${align_dir}/reply
./word-alignment/GIZA++-v2/snt2cooc.out ${align_dir}/post.vcb ${align_dir}/reply.vcb ${align_dir}/post_reply.snt > ${align_dir}/post_reply.cooc
./word-alignment/GIZA++-v2/snt2cooc.out ${align_dir}/reply.vcb ${align_dir}/post.vcb ${align_dir}/reply_post.snt > ${align_dir}/reply_post.cooc
./word-alignment/mkcls-v2/mkcls -p${align_dir}/post -V${align_dir}/post.vcb.classes opt
./word-alignment/mkcls-v2/mkcls -p${align_dir}/reply -V${align_dir}/reply.vcb.classes opt

if [ ! -d "${align_dir}/P2R" ]; then
  mkdir ${align_dir}/P2R
else
  rm ${align_dir}/P2R/*
fi
if [ ! -d "${align_dir}/R2P" ]; then
  mkdir ${align_dir}/R2P
else
  rm ${align_dir}/R2P/*
fi
./word-alignment/GIZA++-v2/GIZA++ -S ${align_dir}/post.vcb -T ${align_dir}/reply.vcb -C ${align_dir}/post_reply.snt -CoocurrenceFile ${align_dir}/post_reply.cooc -outputpath ${align_dir}/P2R/
./word-alignment/GIZA++-v2/GIZA++ -S ${align_dir}/reply.vcb -T ${align_dir}/post.vcb -C ${align_dir}/reply_post.snt -CoocurrenceFile ${align_dir}/reply_post.cooc -outputpath ${align_dir}/R2P/
mv ${align_dir}/R2P/*.actual.ti.final ./data/align_p2r.dict
mv ${align_dir}/P2R/*.actual.ti.final ./data/align_r2p.dict
python ./word-alignment/preprocess.py --pr_input_file $1 --pr_align_file $2 --add_alignment
