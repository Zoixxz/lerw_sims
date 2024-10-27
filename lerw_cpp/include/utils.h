#ifndef UTILS_H
#define UTILS_H

struct Vec2D {
    int x, y;
    bool operator==(const Vec2D& other) const {
        return x == other.x && y == other.y;
    }
};

struct Vec3D {
    int x, y, z;
    bool operator==(const Vec3D& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};

namespace std {
    template <>
    struct hash<Vec2D> {
        std::size_t operator()(const Vec2D& v) const {
            return std::hash<int>()(v.x) ^ std::hash<int>()(v.y << 16);
        }
    };

    template <>
    struct hash<Vec3D> {
        std::size_t operator()(const Vec3D& v) const {
            return std::hash<int>()(v.x) ^ std::hash<int>()(v.y << 16) ^ std::hash<int>()(v.z << 8);
        }
    };
}

#endif // UTILS_H