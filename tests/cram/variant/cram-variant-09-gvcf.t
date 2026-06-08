Candidate-filtered gVCF emits non-variant records for regions with no candidates.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/variant/test-02-supertiny
  > in_bam=${in_dir}/in.aln.bam
  > in_ref=${in_dir}/in.ref.fasta.gz
  > model_var="--model-override ${MODEL_ROOT_DIR}/${MODEL_v600}"
  > touch out/empty_candidates.list
  > ${DORADO_BIN} smallvar --device cpu ${in_bam} ${in_ref} -t 4 ${model_var} --candidate-filtering true --candidates out/empty_candidates.list --regions "chr20:1-100" --ignore-read-groups --gvcf -o out 2> out/stderr
  > echo "Exit code: $?"
  > grep "\[error\]" out/stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/stderr | sed -E 's/.*\[/\[/g'
  > grep -v "^#" out/variants.vcf > out/gvcf.body.vcf
  > wc -l out/gvcf.body.vcf | awk '{ print $1 }'
  > awk 'NR == 1 || NR == 100 { print $1, $2, $4, $5, $7, $9, $10 }' out/gvcf.body.vcf
  Exit code: 0
  [warning] Skipping basecaller compatibility checks for user-specified model override. The accuracy of the results is not guaranteed.
  [warning] Variant calling model is not compatible with the input BAM. This may produce inferior results.
  100
  chr20 1 T . . GT:GQ 0/0:70
  chr20 100 T . . GT:GQ 0/0:70

Candidate-filtered gVCF emits inferred variants and reference records for a targeted region.
The `in.varcall.unsr.list` has the first candidate on position 1614. The variant calling is started at position 1000 to encompass non-inferred regions too.
All candidates in the window: `chr20:1000-2145`:
chr20	1614	2
chr20	1620	2
chr20	1830	4
chr20	1959	1
  $ rm -rf out; mkdir -p out
  > in_dir_1=${TEST_DATA_DIR}/variant/test-02-supertiny
  > in_dir_2=${TEST_DATA_DIR}/variant/test-03-kadayashi-varcall
  > in_bam=${in_dir_1}/in.aln.bam
  > in_ref=${in_dir_1}/in.ref.fasta.gz
  > in_candidates=${in_dir_2}/in.varcall.unsr.list
  > in_expected=${in_dir_2}/expected.gvcf.inferred.chr20_1000_2145.vcf
  > model_var="--model-override ${MODEL_ROOT_DIR}/${MODEL_v600}"
  > ${DORADO_BIN} smallvar --device cpu ${in_bam} ${in_ref} -t 4 ${model_var} --candidate-filtering true --candidates ${in_candidates} --window-len 300 --window-overlap 100 --variant-flanking-bases 100 --regions "chr20:1000-2145" --ignore-read-groups --gvcf -o out 2> out/stderr
  > echo "Exit code: $?"
  > grep "\[error\]" out/stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/stderr | sed -E 's/.*\[/\[/g'
  > grep -v "^#" ${in_expected} | cut -f 1-5,7-10 > out/expected.gvcf.body.no_qual.vcf
  > grep -v "^#" out/variants.vcf | cut -f 1-5,7-10 > out/gvcf.body.no_qual.vcf
  > awk '{ print $1, $2, $3 }' out/processed_regions.bed
  > wc -l out/gvcf.body.no_qual.vcf | awk '{ print $1 }'
  > ### Check the non-variant non-inferred gVCF reference region (GQ is 70)
  > head -n 5 out/gvcf.body.no_qual.vcf | sed -E 's/\t/ /g'
  > ### Check the non-variant reference call from an inferred window (GQ is computed)
  > grep "1615" out/gvcf.body.no_qual.vcf | sed -E 's/\t/ /g'
  > ### Check a variant call
  > grep "1614" out/gvcf.body.no_qual.vcf | sed -E 's/\t/ /g'
  > ### Check that a variant position does not also emit a reference record
  > grep "1959" out/gvcf.body.no_qual.vcf | sed -E 's/\t/ /g'
  Exit code: 0
  [warning] Skipping basecaller compatibility checks for user-specified model override. The accuracy of the results is not guaranteed.
  [warning] Variant calling model is not compatible with the input BAM. This may produce inferior results.
  chr20 1345 1626
  chr20 1534 1812
  chr20 1714 2001
  chr20 1860 2145
  1146
  chr20 1000 . G . . . GT:GQ 0/0:70
  chr20 1001 . C . . . GT:GQ 0/0:70
  chr20 1002 . G . . . GT:GQ 0/0:70
  chr20 1003 . A . . . GT:GQ 0/0:70
  chr20 1004 . C . . . GT:GQ 0/0:70
  chr20 1615 . C . . . GT:GQ 0/0:54
  chr20 1614 . T . . . GT:GQ 0/0:63
  chr20 1959 . T G PASS . GT:GQ 0/1:44

Candidate-filtered gVCF handles a whole-contig region with omitted bounds.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/variant/test-02-supertiny
  > in_bam=${in_dir}/in.aln.bam
  > in_ref=${in_dir}/in.ref.fasta.gz
  > model_var="--model-override ${MODEL_ROOT_DIR}/${MODEL_v600}"
  > touch out/empty_candidates.list
  > ${DORADO_BIN} smallvar --device cpu ${in_bam} ${in_ref} -t 4 ${model_var} --candidate-filtering true --candidates out/empty_candidates.list --regions "chr20" --ignore-read-groups --gvcf -o out 2> out/stderr
  > echo "Exit code: $?"
  > grep "\[error\]" out/stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/stderr | sed -E 's/.*\[/\[/g'
  > grep -v "^#" out/variants.vcf > out/gvcf.body.vcf
  > wc -l out/gvcf.body.vcf | awk '{ print $1 }'
  > awk 'NR == 1 || NR == 10000 { print $1, $2, $4, $5, $7, $9, $10 }' out/gvcf.body.vcf
  Exit code: 0
  [warning] Skipping basecaller compatibility checks for user-specified model override. The accuracy of the results is not guaranteed.
  [warning] Variant calling model is not compatible with the input BAM. This may produce inferior results.
  10000
  chr20 1 T . . GT:GQ 0/0:70
  chr20 10000 C . . GT:GQ 0/0:70

Candidate-filtered gVCF handles a region with an omitted end.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/variant/test-02-supertiny
  > in_bam=${in_dir}/in.aln.bam
  > in_ref=${in_dir}/in.ref.fasta.gz
  > model_var="--model-override ${MODEL_ROOT_DIR}/${MODEL_v600}"
  > touch out/empty_candidates.list
  > ${DORADO_BIN} smallvar --device cpu ${in_bam} ${in_ref} -t 4 ${model_var} --candidate-filtering true --candidates out/empty_candidates.list --regions "chr20:9991" --ignore-read-groups --gvcf -o out 2> out/stderr
  > echo "Exit code: $?"
  > grep "\[error\]" out/stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/stderr | sed -E 's/.*\[/\[/g'
  > grep -v "^#" out/variants.vcf > out/gvcf.body.vcf
  > wc -l out/gvcf.body.vcf | awk '{ print $1 }'
  > awk 'NR == 1 || NR == 10 { print $1, $2, $4, $5, $7, $9, $10 }' out/gvcf.body.vcf
  Exit code: 0
  [warning] Skipping basecaller compatibility checks for user-specified model override. The accuracy of the results is not guaranteed.
  [warning] Variant calling model is not compatible with the input BAM. This may produce inferior results.
  10
  chr20 9991 A . . GT:GQ 0/0:70
  chr20 10000 C . . GT:GQ 0/0:70
