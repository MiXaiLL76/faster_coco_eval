/***************************************************************************
 * Copyright (c) 2019, Martin Renou                                         *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef PYBIND11_JSON_HPP
#define PYBIND11_JSON_HPP

#include <set>
#include <stdexcept>
#include <string>
#include <vector>

// Include nlohmann/json from the same directory
#include "json.hpp"

// Include pybind11 headers
#include <pybind11/pybind11.h>

namespace py = pybind11;
namespace nl = nlohmann;

/**
 * @namespace pyjson
 * @brief Core conversion functions between Python objects and nlohmann::json
 *
 * This namespace contains the main conversion logic that handles all supported
 * Python types and their corresponding JSON representations.
 */
namespace pyjson {
/**
 * @brief Convert nlohmann::json to Python object
 * @param j The JSON object to convert
 * @return Python object representing the JSON data
 *
 * Handles all JSON types: null, bool, numbers, strings, arrays, objects
 * Recursively converts nested structures.
 */
inline py::object from_json(const nl::json& j) {
        if (j.is_null()) {
                return py::none();
        } else if (j.is_boolean()) {
                return py::bool_(j.get<bool>());
        } else if (j.is_number_unsigned()) {
                return py::int_(j.get<nl::json::number_unsigned_t>());
        } else if (j.is_number_integer()) {
                return py::int_(j.get<nl::json::number_integer_t>());
        } else if (j.is_number_float()) {
                return py::float_(j.get<double>());
        } else if (j.is_string()) {
                return py::str(j.get<std::string>());
        } else if (j.is_array()) {
                py::list obj(j.size());
                for (std::size_t i = 0; i < j.size(); i++) {
                        obj[i] = from_json(j[i]);
                }
                return obj;
        } else  // Object
        {
                // Check if this is a special bytes object
                if (j.is_object() && j.contains("__pybind11_bytes__") &&
                    j.size() == 1) {
                        // Restore bytes object from Latin-1 string
                        std::string latin1_str =
                            j["__pybind11_bytes__"].get<std::string>();
                        return py::bytes(latin1_str);
                }

                py::dict obj;
                for (nl::json::const_iterator it = j.cbegin(); it != j.cend();
                     ++it) {
                        obj[py::str(it.key())] = from_json(it.value());
                }
                return obj;
        }
}

/**
 * @brief Convert Python object to nlohmann::json with circular reference
 * detection
 * @param obj The Python object to convert
 * @param refs Set of visited Python object pointers for cycle detection
 * @return JSON representation of the Python object
 * @throws std::runtime_error if circular reference is detected or unsupported
 * type
 *
 * This is the core conversion function that handles all Python → JSON
 * conversion. Memory efficiency: avoids storing Python objects in C++, reduces
 * reference counts.
 */
inline nl::json to_json(const py::handle& obj,
                        std::set<const PyObject*>& refs) {
        if (obj.ptr() == nullptr || obj.is_none()) {
                return nullptr;
        }
        if (py::isinstance<py::bool_>(obj)) {
                return obj.cast<bool>();
        }
        if (py::isinstance<py::int_>(obj)) {
                try {
                        nl::json::number_integer_t s =
                            obj.cast<nl::json::number_integer_t>();
                        if (py::int_(s).equal(obj)) {
                                return s;
                        }
                } catch (...) {
                }
                try {
                        nl::json::number_unsigned_t u =
                            obj.cast<nl::json::number_unsigned_t>();
                        if (py::int_(u).equal(obj)) {
                                return u;
                        }
                } catch (...) {
                }
                throw std::runtime_error(
                    "to_json received an integer out of range for both "
                    "nl::json::number_integer_t and "
                    "nl::json::number_unsigned_t type: " +
                    py::repr(obj).cast<std::string>());
        }

        // Handle NumPy arrays by converting to Python lists
        if (py::hasattr(obj, "tolist")) {
                try {
                        py::object py_list = obj.attr("tolist")();
                        return to_json(py_list, refs);
                } catch (...) {
                        // If tolist() fails, fall through to next check
                }
        }

        // Handle NumPy scalar types by trying to convert them to basic Python
        // types
        if (py::hasattr(obj, "item")) {
                try {
                        py::object item = obj.attr("item")();
                        return to_json(item, refs);
                } catch (...) {
                        // If item() fails, fall through to error
                }
        }
        if (py::isinstance<py::float_>(obj)) {
                return obj.cast<double>();
        }
        if (py::isinstance<py::bytes>(obj)) {
                // Try to decode bytes as Latin-1 string (preserves all byte
                // values 0-255) This is safer than base64 for RLE data that
                // might be used as strings
                try {
                        std::string decoded =
                            obj.attr("decode")("latin-1").cast<std::string>();
                        // Store as an object with special marker to distinguish
                        // from regular strings
                        nl::json bytes_obj;
                        bytes_obj["__pybind11_bytes__"] = decoded;
                        return bytes_obj;
                } catch (...) {
                        // Fallback to base64 if latin-1 fails
                        py::module base64 = py::module::import("base64");
                        return base64.attr("b64encode")(obj)
                            .attr("decode")("utf-8")
                            .cast<std::string>();
                }
        }
        if (py::isinstance<py::str>(obj)) {
                return obj.cast<std::string>();
        }
        if (py::isinstance<py::tuple>(obj) || py::isinstance<py::list>(obj)) {
                auto insert_ret = refs.insert(obj.ptr());
                if (!insert_ret.second) {
                        throw std::runtime_error("Circular reference detected");
                }

                auto out = nl::json::array();
                for (const py::handle value : obj) {
                        out.push_back(to_json(value, refs));
                }

                refs.erase(insert_ret.first);

                return out;
        }
        if (py::isinstance<py::dict>(obj)) {
                auto insert_ret = refs.insert(obj.ptr());
                if (!insert_ret.second) {
                        throw std::runtime_error("Circular reference detected");
                }

                auto out = nl::json::object();
                for (const py::handle key : obj) {
                        // Convert all key types to string for JSON
                        // compatibility
                        std::string key_str;
                        if (py::isinstance<py::str>(key)) {
                                key_str = key.cast<std::string>();
                        } else if (py::isinstance<py::int_>(key)) {
                                key_str = std::to_string(key.cast<long long>());
                        } else if (py::isinstance<py::float_>(key)) {
                                key_str = std::to_string(key.cast<double>());
                        } else {
                                key_str = py::str(key).cast<std::string>();
                        }
                        out[key_str] = to_json(obj[key], refs);
                }

                refs.erase(insert_ret.first);

                return out;
        }

        std::string obj_repr = py::repr(obj).cast<std::string>();
        std::string obj_type = py::str(obj.get_type()).cast<std::string>();
        throw std::runtime_error("to_json not implemented for object type: " +
                                 obj_type + ", repr: " + obj_repr);
}

/**
 * @brief Convert Python object to nlohmann::json (public interface)
 * @param obj The Python object to convert
 * @return JSON representation of the Python object
 *
 * Convenience wrapper that initializes circular reference detection.
 * This is the main entry point for Python → JSON conversion.
 */
inline nl::json to_json(const py::handle& obj) {
        std::set<const PyObject*> refs;
        return to_json(obj, refs);
}

}  // namespace pyjson

/**
 * @namespace nlohmann
 * @brief ADL serializer specializations for automatic conversion
 *
 * These specializations enable automatic conversion between Python objects
 * and nlohmann::json using the Argument Dependent Lookup (ADL) mechanism.
 * This allows writing code like: json j = py_dict; or py::dict d = j;
 */
namespace nlohmann {
/**
 * @brief Macro to create bidirectional serializers for Python types
 * @param T The Python type to create serializers for
 *
 * Creates both to_json and from_json methods for seamless conversion.
 * Used for types that can be safely converted in both directions.
 */
#define MAKE_NLJSON_SERIALIZER_DESERIALIZER(T)                      \
        template <>                                                 \
        struct adl_serializer<T> {                                  \
                inline static void to_json(json& j, const T& obj) { \
                        j = pyjson::to_json(obj);                   \
                }                                                   \
                                                                    \
                inline static T from_json(const json& j) {          \
                        return pyjson::from_json(j);                \
                }                                                   \
        }

/**
 * @brief Macro to create serializer-only for Python handle types
 * @param T The Python type to create serializer for
 *
 * Creates only to_json method for types that should not be deserialized.
 * Used for handles and accessors that are temporary Python references.
 */
#define MAKE_NLJSON_SERIALIZER_ONLY(T)                              \
        template <>                                                 \
        struct adl_serializer<T> {                                  \
                inline static void to_json(json& j, const T& obj) { \
                        j = pyjson::to_json(obj);                   \
                }                                                   \
        }

// Core Python object types - bidirectional conversion
MAKE_NLJSON_SERIALIZER_DESERIALIZER(py::object);

// Basic Python types - bidirectional conversion
MAKE_NLJSON_SERIALIZER_DESERIALIZER(py::bool_);
MAKE_NLJSON_SERIALIZER_DESERIALIZER(py::int_);
MAKE_NLJSON_SERIALIZER_DESERIALIZER(py::float_);
MAKE_NLJSON_SERIALIZER_DESERIALIZER(py::str);

// Container types - bidirectional conversion
MAKE_NLJSON_SERIALIZER_DESERIALIZER(py::list);
MAKE_NLJSON_SERIALIZER_DESERIALIZER(py::tuple);
MAKE_NLJSON_SERIALIZER_DESERIALIZER(
    py::dict);  // This is the key one for Dataset optimization!

// Handle and accessor types - serialization only (temporary references)
MAKE_NLJSON_SERIALIZER_ONLY(py::handle);
MAKE_NLJSON_SERIALIZER_ONLY(py::detail::item_accessor);
MAKE_NLJSON_SERIALIZER_ONLY(py::detail::list_accessor);
MAKE_NLJSON_SERIALIZER_ONLY(py::detail::tuple_accessor);
MAKE_NLJSON_SERIALIZER_ONLY(py::detail::sequence_accessor);
MAKE_NLJSON_SERIALIZER_ONLY(py::detail::str_attr_accessor);
MAKE_NLJSON_SERIALIZER_ONLY(py::detail::obj_attr_accessor);

#undef MAKE_NLJSON_SERIALIZER
#undef MAKE_NLJSON_SERIALIZER_ONLY
}  // namespace nlohmann

/**
 * @namespace pybind11::detail
 * @brief Type caster for nlohmann::json
 *
 * This enables nlohmann::json to be used directly as function parameters
 * and return types in pybind11 function bindings. The caster automatically
 * converts between Python objects and JSON at the function call boundary.
 *
 * Memory efficiency: Conversion happens only at Python/C++ boundary,
 * allowing efficient JSON storage in C++ with automatic Python interop.
 */
namespace pybind11 {
namespace detail {
template <>
struct type_caster<nl::json> {
       public:
        PYBIND11_TYPE_CASTER(nl::json, _("json"));

        /**
         * @brief Convert Python object to JSON (Python → C++)
         * @param src Python object handle
         * @param unused Conversion flags (unused)
         * @return true if conversion successful, false otherwise
         */
        bool load(handle src, bool) {
                try {
                        value = pyjson::to_json(src);
                        return true;
                } catch (...) {
                        return false;
                }
        }

        /**
         * @brief Convert JSON to Python object (C++ → Python)
         * @param src JSON object to convert
         * @param policy Return value policy (unused)
         * @param parent Parent object (unused)
         * @return Python object handle
         */
        static handle cast(nl::json src, return_value_policy /* policy */,
                           handle /* parent */) {
                object obj = pyjson::from_json(src);
                return obj.release();
        }
};
}  // namespace detail
}  // namespace pybind11

#endif
