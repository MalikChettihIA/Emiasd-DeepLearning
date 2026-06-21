# Compilateur et flags
CXX := clang++
CXXFLAGS := -O3 -Wall -shared -std=c++17 -fvisibility=hidden -undefined dynamic_lookup
DEPLOYMENT_TARGET := 13.0

# Chemins Python
PYTHON_INCLUDES := $(shell python3 -m pybind11 --includes) $(shell python3-config --includes)
EXTENSION_SUFFIX := $(shell python3-config --extension-suffix)

# Fichiers sources
SRC := golois.cpp
HEADERS := Board.h Game.h Rzone.h
TARGET := golois$(EXTENSION_SUFFIX)

all: $(TARGET)

$(TARGET): $(SRC) $(HEADERS)
	export MACOSX_DEPLOYMENT_TARGET=$(DEPLOYMENT_TARGET); \
	$(CXX) $(CXXFLAGS) $(PYTHON_INCLUDES) $< -o $@

clean:
	rm -f *.so *.dylib

.PHONY: all clean
