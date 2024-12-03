# import math

# def sum_of_two_squares(n):
#     if n == 1:
#         return True

#     while n % 2 == 0:
#         n = n // 2

#     for p in range(3, int(math.sqrt(n)) + 1, 4):
#         if n % p == 0:
#             count = 0
#             while n % p == 0:
#                 n //= p
#                 count += 1
#             if count % 2 != 0:
#                 return False
#     if n > 1 and n % 4 == 3:
#         return False

#     return True

# x = 1
# while True:
#     if x % 10000 == 0:
#         print(math.isqrt(x))
#     if sum_of_two_squares(x) and math.isqrt(x) > (2**15):
#         break
#     x += 1

# print(x)

with open('/home/epsilon/Code/Research/Long_Range_Simulations/lerw_sims/lerw_2d/valid_nums.txt', 'r') as file:
        lines = file.readlines()

# Retain only the first `keep_lines` lines
truncated_lines = lines[:48168466]

# Write the truncated content back to the file
with open('/home/epsilon/Code/Research/Long_Range_Simulations/lerw_sims/lerw_2d/valid_nums2.txt', 'w') as file:
    file.writelines(truncated_lines)