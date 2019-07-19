MODULES = modules/
OBJECTS = modules/
PRGNAME = cbrs
DEPENDS = $(OBJECTS)cb.o          \
          $(OBJECTS)file.o        \
          $(OBJECTS)interfc.o     \
          $(OBJECTS)memctrl.o     \
          $(OBJECTS)random.o      \
          $(OBJECTS)reporting.o   \
          $(OBJECTS)rs.o
	  
OPT     = -O3 -Wall -lm -I. -I$(MODULES)

all: $(PRGNAME)

$(PRGNAME): $(PRGNAME).o $(DEPENDS) 
	gcc -o $(PRGNAME) $(OPT) $(PRGNAME).o $(DEPENDS) 

$(PRGNAME).o: $(PRGNAME).c
	gcc $(OPT) -c $(PRGNAME).c -o $(PRGNAME).o

$(OBJECTS)%.o: %.c
	gcc $(OPT) -c $< -o $@

$(OBJECTS)%.o: $(MODULES)%.c
	gcc $(OPT) -c $< -o $@

.PHONY : clean
clean: 
	rm $(DEPENDS) $(PRGNAME).o

