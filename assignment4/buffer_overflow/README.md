# Buffer Overflow Exercises


A buffer overflow happens when a function writes more data to a buffer than it can accommodate, often as a result of not checking bounds. During execution, this allows an attacker to overwrite saved processor registers and return addresses, enabling the modification of program control flow, which can be exploited maliciously. These exercises demonstrate how such an attack would take place.

---

## Program Execution

Software developers often program in high-level languages like C or C++. Even to experienced developers, it is often unclear how these programs go from being text files to actually executing on a CPU. This section very briefly explains how that occurs.

Take this simple C program:

```c
#include <stdio.h>

int global_counter = 0;
const char* greeting = "hello";

int main() {
    int x = 42;
    char buf[16];
    printf("x is at %p\n", &x);
    return 0;
}
```

After compilation with `gcc`, the resulting binary is not a flat blob of machine code. Instead, it is divided into named sections, each with an explicit purpose and set of memory permissions. The below table shows some of the sections in a typical "ELF" executable.

| Section | Contents | Permissions |
| ------- | -------- | ----------- |
| `.text` | Compiled machine instructions | Read + Execute |
| `.rodata` | Read-only data (string literals, `const` globals) | Read only |
| `.data` | Initialised global/static variables | Read + Write |
| `.bss` | Uninitialised global/static variables (zero-filled at load time) | Read + Write |

Separating code from data matters for both performance and security. For instance, by storing all compiled instructions in `.text`, the CPU can reliably execute that region without needing to check whether each byte is data or code. Further, the separate permission bits mean that `.text` cannot be written (preventing code injection into `.text`). Additionally, `.bss`, `.data` are not marked executable, eliminating the ability to inject malicious code to those regions and subsequently jumping to them. 

### The Virtual Address Space

Once the executable has been prepared into the above sections, Linux loads the program into memory at fixed virtual addresses, with two more regions (the stack and heap) that exist only at runtime:

```
0xFFFFFFFFFFFFFFFF  +-----------------------+
                    |       kernel          |
0x7FFFFFFFFFFF      +-----------------------+
                    |       stack           |  grows downward
                    |          |            |
                    |          v            |
                    |                       |
                    |          ^            |
                    |          |            |
                    |        heap           |  grows upward
                    +-----------------------+
                    |    .bss / .data       |
                    +-----------------------+
                    |       .rodata         |
                    +-----------------------+
0x0000000000400000  |       .text           |
                    +-----------------------+
```

To then execute the program, Linux essentially loads the program into memory, sets the Program Counter to 0x40_0000 (the .text region) and starts executing. For the purposes of this lab, we ignore the heap entirely, as all our variables are local and statically allocated, placing them in the stack, primarily by the compiler. 

The compiler determines at compile time how many bytes each function needs on the stack, and reserves that space at the start of the function by decrementing the stack pointer. All local variables (ints, floats, arrays, pointers) live in that reserved region. Each variable is accessed via a fixed offset from the base pointer, not by moving the stack pointer around.

---

## The Call Stack

As mentioned, the stack contains all function data and `.text` contains all instructions. When the CPU executes a `call` instruction it:

1. Pushes the **return address** (the address of the instruction immediately after the `call`) onto the stack.
2. Jumps to the first instruction of the called function.

The called function's **prologue** then:

1. Pushes the caller's **base pointer** (`RBP`) onto the stack to save it.
2. Sets `RBP` to the current stack pointer, establishing a stable reference point for this frame.
3. Decrements `RSP` to reserve space for local variables.

The result is a stack frame that looks like this:

```
Higher addresses (toward top of address space)
+---------------------------+
|       return address      |  <-- where to go when the function returns
+---------------------------+
|       saved RBP           |  <-- caller's base pointer
+---------------------------+
|    local variables        |
+---------------------------+
|        buf[16]            |  <-- arrays live here
+---------------------------+
Writing to buf moves stack pointer UPWARDS
```

Remember, each function has an associated .text section and stack region for data. The above figure depicts a given functions's stack region. On return, the CPU restores `RBP` and loads the return address into the instruction pointer `RIP`, resuming the caller exactly where it left off.

### The critical layout fact

During allocation of stack space, the stack grows **downward**, by decrementing the stack pointer (0x7fff...). Naturally, when subsequently accessing a stack pointer during execution, the buffer grows upwards (i.e., buf[1] is at a larger address than buf[0], which means its at a "higher" location). Writing past the end of `buf` therefore writes into higher-addressed memory, directly toward the saved `RBP` and then the return address.

If an attacker controls what gets written past the end of `buf`, they control where the function returns.

---

## Why `strcpy` is Dangerous

`strcpy(dest, src)` copies bytes from `src` to `dest` until it encounters a null byte `\0`. It does **not** check whether `dest` is large enough. If `src` is longer than `dest`, `strcpy` keeps incrementing both pointers and writing, overwriting whatever sits above `dest` in the stack frame.

Writing the address of an arbitrary function at exactly the right offset makes the function return there instead of back to its caller. This is the fundemental operating principle of buffer-overwrite attacks. 

---

## What is a Segmentation Fault?

While working through these exercises, you will frequently see the program crash with a message like:

```
Segmentation fault (core dumped)
```

A **segmentation fault** (often shortened to "segfault" or "SIGSEGV") is the operating system's way of telling you that your program tried to access memory in a way it was not allowed to. As described above, each region of the virtual address space has a defined set of permissions (read, write, execute) and a defined range of valid addresses. When the CPU attempts an access that violates those rules, the hardware raises an exception, the kernel catches it, and it terminates the offending process.

Common causes include:

- **Dereferencing an invalid pointer**, such as a `NULL` pointer or a pointer to memory that has already been freed.
- **Writing to a read-only region**, for example trying to modify a string literal stored in `.rodata`.
- **Executing non-executable memory**, such as jumping into `.data` or the stack when those regions are not marked executable.
- **Jumping to an invalid address**, which is exactly what happens when a buffer overflow corrupts the return address with garbage. When the function returns, `RIP` is loaded with a meaningless value, the CPU tries to fetch an instruction from an unmapped or non-executable page, and the program segfaults.


