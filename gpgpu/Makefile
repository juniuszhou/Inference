# GPGPU CUDA Project Makefile

CUDA      ?= nvcc
CUDAFLAGS ?= -arch=sm_86 -O2
NCCLFLAGS ?= -I/usr/include -L/usr/lib/x86_64-linux-gnu -lnccl
CUBLASFLAGS ?= -lcublas

BUILDDIR  = build
TESTDIR   = tests
SRCDIR    = src
FLASHDIR  = flash-attention-minimal

# Standalone executable (entry point)
EXES = $(BUILDDIR)/main

# Test executables
TESTS = $(TESTDIR)/test_add \
        $(TESTDIR)/test_multiply \
        $(TESTDIR)/test_shmem \
        $(TESTDIR)/test_nccl \
        $(TESTDIR)/test_basic \
        $(TESTDIR)/test_blas

.PHONY: all _build clean test test_add test_multiply test_shmem test_nccl test_basic test_blas run run-main run-flash help

all: _build

_build: $(EXES)
	@echo "Build complete."

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

$(TESTDIR):
	mkdir -p $(TESTDIR)

# Kernel objects for tests
$(BUILDDIR)/add_test.o: $(SRCDIR)/add.cu | $(BUILDDIR)
	$(CUDA) $(CUDAFLAGS) -c $< -o $@

$(BUILDDIR)/multiply_test.o: $(SRCDIR)/multiply.cu | $(BUILDDIR)
	$(CUDA) $(CUDAFLAGS) -c $< -o $@

$(BUILDDIR)/shmem_test.o: $(SRCDIR)/shmem.cu | $(BUILDDIR)
	$(CUDA) $(CUDAFLAGS) -c $< -o $@

$(BUILDDIR)/basic_test.o: $(SRCDIR)/basic.cu | $(BUILDDIR)
	$(CUDA) $(CUDAFLAGS) -c $< -o $@

$(BUILDDIR)/nccl_test.o: $(SRCDIR)/nccl_demo.cu | $(BUILDDIR)
	$(CUDA) $(CUDAFLAGS) $(NCCLFLAGS) -c $< -o $@

# Standalone executable
$(BUILDDIR)/main: main.cu $(BUILDDIR)/basic.o | $(BUILDDIR)
	$(CUDA) $(CUDAFLAGS) $< $(BUILDDIR)/basic.o -o $@

$(BUILDDIR)/basic.o: $(SRCDIR)/basic.cu | $(BUILDDIR)
	$(CUDA) $(CUDAFLAGS) -c $< -o $@


# Test executables
$(TESTDIR)/test_add: $(TESTDIR)/test_add.cu $(BUILDDIR)/add_test.o | $(TESTDIR)
	$(CUDA) $(CUDAFLAGS) $< $(BUILDDIR)/add_test.o -o $@

$(TESTDIR)/test_basic: $(TESTDIR)/test_basic.cu $(BUILDDIR)/basic_test.o | $(TESTDIR)
	$(CUDA) $(CUDAFLAGS) $< $(BUILDDIR)/basic_test.o -o $@

$(TESTDIR)/test_multiply: $(TESTDIR)/test_multiply.cu $(BUILDDIR)/multiply_test.o | $(TESTDIR)
	$(CUDA) $(CUDAFLAGS) $< $(BUILDDIR)/multiply_test.o -o $@

$(TESTDIR)/test_shmem: $(TESTDIR)/test_shmem.cu $(BUILDDIR)/shmem_test.o | $(TESTDIR)
	$(CUDA) $(CUDAFLAGS) $< $(BUILDDIR)/shmem_test.o -o $@

$(TESTDIR)/test_nccl: $(TESTDIR)/test_nccl.cu $(BUILDDIR)/nccl_test.o | $(TESTDIR)
	$(CUDA) $(CUDAFLAGS) $< $(BUILDDIR)/nccl_test.o $(NCCLFLAGS) -o $@

$(TESTDIR)/test_blas: $(TESTDIR)/test_blas.cu | $(TESTDIR)
	$(CUDA) $(CUDAFLAGS) $< $(CUBLASFLAGS) -o $@

clean:
	rm -rf $(BUILDDIR)
	rm -f $(TESTS)
	@echo "Cleaned build artifacts."

test: _build $(TESTS)
	@echo "========================================"
	@echo "Running CUDA unit tests"
	@echo "========================================"
	@for test in $(TESTS); do \
		echo ""; \
		echo ">>> Running $$test"; \
		echo "----------------------------------------"; \
		./$$test || exit 1; \
	done
	@echo ""
	@echo "========================================"
	@echo "All CUDA unit tests PASSED"
	@echo "========================================"
	rm -f $(TESTS)
	@echo "Cleaned test binaries."

# Individual tests (self-contained: build -> run -> clean)
test_add: $(BUILDDIR)/add_test.o
	$(CUDA) $(CUDAFLAGS) $(TESTDIR)/test_add.cu $(BUILDDIR)/add_test.o -o $(TESTDIR)/test_add
	./$(TESTDIR)/test_add
	rm -f $(TESTDIR)/test_add

test_basic: $(BUILDDIR)/basic_test.o
	$(CUDA) $(CUDAFLAGS) $(TESTDIR)/test_basic.cu $(BUILDDIR)/basic_test.o -o $(TESTDIR)/test_basic
	./$(TESTDIR)/test_basic
	rm -f $(TESTDIR)/test_basic

test_multiply: $(BUILDDIR)/multiply_test.o
	$(CUDA) $(CUDAFLAGS) $(TESTDIR)/test_multiply.cu $(BUILDDIR)/multiply_test.o -o $(TESTDIR)/test_multiply
	./$(TESTDIR)/test_multiply
	rm -f $(TESTDIR)/test_multiply

test_shmem: $(BUILDDIR)/shmem_test.o
	$(CUDA) $(CUDAFLAGS) $(TESTDIR)/test_shmem.cu $(BUILDDIR)/shmem_test.o -o $(TESTDIR)/test_shmem
	./$(TESTDIR)/test_shmem
	rm -f $(TESTDIR)/test_shmem

test_nccl: $(BUILDDIR)/nccl_test.o
	$(CUDA) $(CUDAFLAGS) $(TESTDIR)/test_nccl.cu $(BUILDDIR)/nccl_test.o $(NCCLFLAGS) -o $(TESTDIR)/test_nccl
	./$(TESTDIR)/test_nccl
	rm -f $(TESTDIR)/test_nccl

test_blas:
	$(CUDA) $(CUDAFLAGS) $(TESTDIR)/test_blas.cu $(CUBLASFLAGS) -o $(TESTDIR)/test_blas
	./$(TESTDIR)/test_blas
	rm -f $(TESTDIR)/test_blas

run: _build
	@echo "=== Running main ==="
	./$(BUILDDIR)/main

run-main: _build
	./$(BUILDDIR)/main

run-flash:
	@echo "=== Running flash attention benchmark (requires PyTorch) ==="
	python3 $(FLASHDIR)/bench.py

help:
	@echo "Available targets:"
	@echo "  _build        - Compile all CUDA examples"
	@echo "  test          - Build and run all CUDA unit tests (clean binaries after)"
	@echo "  test_add      - Run single test: add.cu kernels"
	@echo "  test_multiply - Run single test: multiply.cu kernels"
	@echo "  test_shmem    - Run single test: shmem.cu kernels"
	@echo "  test_basic    - Run single test: basic.cu kernels"
	@echo "  test_nccl     - Run single test: nccl_demo.cu (broadcast/reduce/allreduce)"
	@echo "  test_blas     - Run single test: cuBLAS SGEMM"
	@echo "  run_main   - Build and run main.cu"
	@echo "  clean         - Remove build artifacts"
	@echo "  run-flash     - Run flash attention benchmark (needs PyTorch)"
	@echo "  help          - Show this help"
