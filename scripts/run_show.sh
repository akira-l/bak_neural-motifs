echo "run relation presentation"
python models/count_rels.py -m sgcls -model motifnet -order leftright -nl_obj 2 -nl_edge 4 -b 1 -clip 5  -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -test -ckpt checkpoints/motifnet-sgcls/vgrel-motifnet-sgcls.tar -nepoch 50 -use_bias -cache motifnet_sgcls

#python models/show_rels.py -m predcls -model motifnet -order leftright -nl_obj 2 -nl_edge 4 -b 1 -clip 5  -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -test -ckpt checkpoints/motifnet-sgcls/vgrel-motifnet-sgcls.tar -nepoch 50 -use_bias -cache motifnet_sgcls


#python models/show_rels.py -m sgdet -model motifnet -order leftright -nl_obj 2 -nl_edge 4 -b 1 -clip 5  -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -test -ckpt checkpoints/motifnet-sgdet/vgrel-motifnet-sgdet.tar -nepoch 50 -use_bias -cache motifnet_sgdet.pkl
