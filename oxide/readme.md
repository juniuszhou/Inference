# oxide usage


## install 
```bash
cargo install --git https://github.com/NVlabs/cuda-oxide.git cargo-oxide
```

## build 
```bash
cargo oxide build
```

## run
```bash
cargo oxide run
```

## test
```bash
cargo oxide test

# test a file
cargo oxide test -- --package first --test norm

# test a case
cargo oxide test -- --package first --test norm -- test_rmsnorm --exact

```

## get the ptx for all functions
```bash
# touch it then we can compile it again
touch src/lib.rs

# build it get first ptx
cargo oxide build -- --lib

# check the file
ls -l first.ptx
```


