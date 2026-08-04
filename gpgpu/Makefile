# GPGPU CUDA Project Makefile

CUDA      ?= nvcc
CUDAFLAGS ?= -arch=sm_86 -O2
CXX       ?= g++
CXXFLAGS  ?= -std=c++17

BUILDDIR  = build
TESTDIR   = tests
SRCDIR    = src
FLASHDIR  = flash-attention-minimal

# Standalone executables (have main)
EXES = $(BUILDDIR)/main $(BUILDDIR)/basic $(BUILDDIR)/shmem

# Kernel-only .cu files (no main) - built as .o for potential reuse
KERN_OBJS = $(BUILDDIR)/add.o $(BUILDDIR)/multiply.o

# Build targets with no main (compiled but not linked)
NO_MAIN_EXES = $(BUILDDIR)/nccl_demo

# Test executables
TESTS = $(TESTDIR)/test_add \
        $(TESTDIR)/test_multiply \
        $(TESTDIR)/test_shmem \
        $(TESTDIR)/test_nccl

.PHONY: all _build clean test run run-flash help

all: _build

_build: $(EXES) $(KERN_OBJS) $(NO_MAIN_EXES)
	@echo "Build complete."

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

$(BUILDDIR)/main: main.cu | $(BUILDDIR)
	$(CUDA) $(CUDAFLAGS) $< -o $@

$(BUILDDIR)/basic: $(SRCDIR)/basic.cu | $(BUILDDIR)
	$(CUDA) $(CUDAFLAGS) $< -o $@

$(BUILDDIR)/shmem: $(SRCDIR)/shmem.cu | $(BUILDDIR)
	$(CUDA) $(CUDAFLAGS) $< -o $@

$(BUILDDIR)/nccl_demo: $(SRCDIR)/nccl_demo.cu | $(BUILDDIR)
	$(CUDA) $(CUDAFLAGS) -c $< -o $@

$(BUILDDIR)/add.o: $(SRCDIR)/add.cu | $(BUILDDIR)
	$(CUDA) $(CUDAFLAGS) -c $< -o $@

$(BUILDDIR)/multiply.o: $(SRCDIR)/multiply.cu | $(BUILDDIR)
	$(CUDA) $(CUDAFLAGS) -c $< -o $@

$(TESTDIR):
	mkdir -p $(TESTDIR)

$(TESTDIR)/test_add: $(TESTDIR)/test_add.cu | $(TESTDIR)
	$(CUDA) $(CUDAFLAGS) $< -o $@

$(TESTDIR)/test_multiply: $(TESTDIR)/test_multiply.cu | $(TESTDIR)
	$(CUDA) $(CUDAFLAGS) $< -o $@

$(TESTDIR)/test_shmem: $(TESTDIR)/test_shmem.cu | $(TESTDIR)
	$(CUDA) $(CUDAFLAGS) $< -o $@

$(TESTDIR)/test_nccl: $(TESTDIR)/test_nccl.cu | $(TESTDIR)
	$(CUDA) $(CUDAFLAGS) $< -o $@

clean:
	rm -rf $(BUILDDIR)
	rm -f $(TESTDIR)/test_add $(TESTDIR)/test_multiply $(TESTDIR)/test_shmem $(TESTDIR)/test_nccl
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

run: _build
	@echo "=== Running main ==="
	./$(BUILDDIR)/main
	@echo "=== Running basic ==="
	./$(BUILDDIR)/basic
	@echo "=== Running shmem ==="
	./$(BUILDDIR)/shmem

run-flash:
	@echo "=== Running flash attention benchmark (requires PyTorch) ==="
	python3 $(FLASHDIR)/bench.py

help:
	@echo "Available targets:"
	@echo "  _build     - Compile all CUDA examples"
	@echo "  test       - Build and run all CUDA unit tests (clean binaries after)"
	@echo "  run        - Build and run standalone CUDA demos"
	@echo "  clean      - Remove build artifacts"
	@echo "  run-flash  - Run flash attention benchmark (requires PyTorch)"
	@echo "  help       - Show this help"
	@echo ""
	@echo "Options:"
	@echo "  CUDA=nvcc       - Set CUDA compiler (default: nvcc)"
	@echo "  CUDAFLAGS=-arch=sm_86 - Set CUDA architecture flags"
