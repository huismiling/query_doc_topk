#sh run.sh

set -x

# ROOT_DIR=$(cd $(dirname $0); pwd)
ROOT_DIR=$1
query_dir=$2
doc_file=$3
output_file=$4

if [ -z "$ROOT_DIR" ]; then
        ROOT_DIR=$(cd $(dirname $0); pwd)
fi

if [ -z "$doc_file" ]; then
        doc_file=./translate/docs.txt
fi

if [ -z "$output_file" ]; then
        output_file=./translate/res/result.txt
fi

if [ -z "$query_dir" ]; then
        query_dir=./translate/querys
fi

$ROOT_DIR/bin/query_doc_scoring ${doc_file} ${query_dir} ${output_file} 1100
# $ROOT_DIR/bin/query_doc_scoring ${doc_file} ${query_dir} ${output_file} 8192

# p_num=512
# nsys profile -o report_nquerys_1024_p_${p_num}k.qdrep $ROOT_DIR/bin/query_doc_scoring ${doc_file} ${query_dir} ${output_file} 1024
# nsys profile -o report_nquerys_2048_p_${p_num}k.qdrep $ROOT_DIR/bin/query_doc_scoring ${doc_file} ${query_dir} ${output_file} 2048
# nsys profile -o report_nquerys_4096_p_${p_num}k.qdrep $ROOT_DIR/bin/query_doc_scoring ${doc_file} ${query_dir} ${output_file} 4096
# nsys profile -o report_nquerys_8192_p_${p_num}k.qdrep $ROOT_DIR/bin/query_doc_scoring ${doc_file} ${query_dir} ${output_file} 8192
echo "run success"
