TARGET = pcl_view
CONFIG += console
CONFIG -= app_bundle
CONFIG += link_pkgconfig
PKGCONFIG += eigen3
PKGCONFIG += pcl_io-1.7
QMAKE_CXXFLAGS += -std=c++11

INCLUDEPATH += /usr/include/pcl-1.7
INCLUDEPATH += /usr/include/vtk-6.2

LIBS += -L/usr/lib/ -lboost_system
LIBS += `pkg-config \
    --cflags \
    --libs`

LIBS += -lpcl_visualization \

SOURCES += main.cpp \



