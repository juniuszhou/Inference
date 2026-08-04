# GPGPU CUDA Project Makefile

CUDA      ?= nvcc
CUDAFLAGS ?= -arch=sm_86 -O2
NCCLFLAGS ?= -I/usr/include -L/usr/lib/x86_64-linux-gnu -lnccl

BUILDDIR  = build
TESTDIR   = tests
SRCDIR    = src
FLASHDIR  = flash-attention-minimal

# Standalone executables
EXES = $(BUILDDIR)/main $(BUILDDIR)/basic $(BUILDDIR)/shmem $(BUILDDIR)/nccl_demo

# Test executables
TESTS = $(TESTDIR)/test_add \
        $(TESTDIR)/test_multiply \
        $(TESTDIR)/test_shmem \
        $(TESTDIR)/test_nccl

.PHONY: all _build clean test run run-flash help

all: _build

_build: $(EXES)
	@echo "Build complete."

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

$(TESTDIR):
	mkdir -p $(TESTDIR)

# Kernel objects for tests (NO_MAIN defined)
$(BUILDDIR)/add_test.o: $(SRCDIR)/add.cu | $(BUILDDIR)
	$(CUDA) $(CUDAFLAGS) -DNO_MAIN -c $< -o $@

$(BUILDDIR)/multiply_test.o: $(SRCDIR)/multiply.cu | $(BUILDDIR)
	$(CUDA) $(CUDAFLAGS) -DNO_MAIN -c $< -o $@

$(BUILDDIR)/shmem_test.o: $(SRCDIR)/shmem.cu | $(BUILDDIR)
	$(CUDA) $(CUDAFLAGS) -DNO_MAIN -c $< -o $@

$(BUILDDIR)/nccl_test.o: $(SRCDIR)/nccl_demo.cu | $(BUILDDIR)
	$(CUDA) $(CUDAFLAGS) -DNO_MAIN $(NCCLFLAGS) -c $< -o $@

# Standalone executables
$(BUILDDIR)/main: main.cu $(BUILDDIR)/basic_no_main.o | $(BUILDDIR)
	$(CUDA) $(CUDAFLAGS) $< $(BUILDDIR)/basic_no_main.o -o $@

$(BUILDDIR)/basic_no_main.o: $(SRCDIR)/basic.cu | $(BUILDDIR)
	$(CUDA) $(CUDAFLAGS) -DNO_MAIN -c $< -o $@

$(BUILDDIR)/basic: $(SRCDIR)/basic.cu | $(BUILDDIR)
	$(CUDA) $(CUDAFLAGS) $< -o $@

$(BUILDDIR)/shmem: $(SRCDIR)/shmem.cu | $(BUILDDIR)
	$(CUDA) $(CUDAFLAGS) $< -o $@

$(BUILDDIR)/nccl_demo: $(SRCDIR)/nccl_demo.cu | $(BUILDDIR)
	$(CUDA) $(CUDAFLAGS) $< $(NCCLFLAGS) -o $@

# Test executables
$(TESTDIR)/test_add: $(TESTDIR)/test_add.cu $(BUILDDIR)/add_test.o | $(TESTDIR)
	$(CUDA) $(CUDAFLAGS) $< $(BUILDDIR)/add_test.o -o $@

$(TESTDIR)/test_multiply: $(TESTDIR)/test_multiply.cu $(BUILDDIR)/multiply_test.o | $(TESTDIR)
	$(CUDA) $(CUDAFLAGS) $< $(BUILDDIR)/multiply_test.o -o $@

$(TESTDIR)/test_shmem: $(TESTDIR)/test_shmem.cu $(BUILDDIR)/shmem_test.o | $(TESTDIR)
	$(CUDA) $(CUDAFLAGS) $< $(BUILDDIR)/shmem_test.o -o $@

$(TESTDIR)/test_nccl: $(TESTDIR)/test_nccl.cu $(BUILDDIR)/nccl_test.o | $(TESTDIR)
	$(CUDA) $(CUDAFLAGS) $< $(BUILDDIR)/nccl_test.o $(NCCLFLAGS) -o $@

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

test_add: $(TESTDIR)/test_add
	./$(TESTDIR)/test_add
	rm -f $(TESTDIR)/test_add

test_multiply: $(TESTDIR)/test_multiply
	./$(TESTDIR)/test_multiply
	rm -f $(TESTDIR)/test_multiply

test_shmem: $(TESTDIR)/test_shmem
	./$(TESTDIR)/test_shmem
	rm -f $(TESTDIR)/test_shmem

test_nccl: $(TESTDIR)/test_nccl
	./$(TESTDIR)/test_nccl
	rm -f $(TESTDIR)/test_nccl

run: _build
	@echo "=== Running main ==="
	./$(BUILDDIR)/main
	@echo "=== Running basic ==="
	./$(BUILDDIR)/basic
	@echo "=== Running shmem ==="
	./$(BUILDDIR)/shmem
	@echo "=== Running nccl_demo ==="
	./$(BUILDDIR)/nccl_demo

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
	@echo "  test_nccl     - Run single test: nccl_demo.cu (broadcast/reduce/allreduce)"
	@echo "  run_main   - Build and run main.cu"
	@echo "  clean         - Remove build artifacts"
	@echo "  run-flash     - Run flash attention benchmark (needs PyTorch)"
	@echo "  help          - Show this help"
