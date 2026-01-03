#pragma once

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace taco::expr {

// Small, dependency-free expression compiler for scalar functions f(w).
// Intended use: parse J(w) from config files.
//
// Supported:
// - variable: w
// - constants: pi, e
// - operators: + - * / ^
// - functions (1 arg): exp, log, sqrt, abs, sin, cos, tan, step
// - functions (2 args): pow, max, min
//
// Notes:
// - Identifiers other than w/pi/e must be substituted externally before compiling.
// - No implicit multiplication (write `a*b`, not `ab`).
class Expression {
  public:
    Expression() = default;

    static Expression compile(std::string source,
                              const std::vector<std::pair<std::string, double>>& params = {}) {
        Expression out;
        out.source_ = std::move(source);
        out.rpn_ = compile_to_rpn(out.source_, params);
        return out;
    }

    double eval(double w) const {
        if (rpn_.empty()) throw std::runtime_error("Expression is empty");
        std::vector<double> st;
        st.reserve(rpn_.size());
        for (const auto& t : rpn_) {
            switch (t.kind) {
                case Kind::Number:
                    st.push_back(t.value);
                    break;
                case Kind::VarW:
                    st.push_back(w);
                    break;
                case Kind::Op1: {
                    if (st.empty()) throw std::runtime_error("Expression stack underflow (unary)");
                    double a = st.back();
                    st.pop_back();
                    if (t.op == Op::Neg) st.push_back(-a);
                    else throw std::runtime_error("Unsupported unary op");
                    break;
                }
                case Kind::Op2: {
                    if (st.size() < 2) throw std::runtime_error("Expression stack underflow (binary)");
                    const double b = st.back();
                    st.pop_back();
                    const double a = st.back();
                    st.pop_back();
                    switch (t.op) {
                        case Op::Add: st.push_back(a + b); break;
                        case Op::Sub: st.push_back(a - b); break;
                        case Op::Mul: st.push_back(a * b); break;
                        case Op::Div: st.push_back(a / b); break;
                        case Op::Pow: st.push_back(std::pow(a, b)); break;
                        default: throw std::runtime_error("Unsupported binary op");
                    }
                    break;
                }
                case Kind::Func1: {
                    if (st.empty()) throw std::runtime_error("Expression stack underflow (func1)");
                    const double a = st.back();
                    st.pop_back();
                    switch (t.fn) {
                        case Fn::Exp: st.push_back(std::exp(a)); break;
                        case Fn::Log: st.push_back(std::log(a)); break;
                        case Fn::Sqrt: st.push_back(std::sqrt(a)); break;
                        case Fn::Abs: st.push_back(std::abs(a)); break;
                        case Fn::Sin: st.push_back(std::sin(a)); break;
                        case Fn::Cos: st.push_back(std::cos(a)); break;
                        case Fn::Tan: st.push_back(std::tan(a)); break;
                        case Fn::Step: st.push_back(a > 0.0 ? 1.0 : 0.0); break;
                        default: throw std::runtime_error("Unsupported func1");
                    }
                    break;
                }
                case Kind::Func2: {
                    if (st.size() < 2) throw std::runtime_error("Expression stack underflow (func2)");
                    const double b = st.back();
                    st.pop_back();
                    const double a = st.back();
                    st.pop_back();
                    switch (t.fn) {
                        case Fn::Pow2: st.push_back(std::pow(a, b)); break;
                        case Fn::Max: st.push_back(std::max(a, b)); break;
                        case Fn::Min: st.push_back(std::min(a, b)); break;
                        default: throw std::runtime_error("Unsupported func2");
                    }
                    break;
                }
            }
        }
        if (st.size() != 1) throw std::runtime_error("Expression did not reduce to a scalar");
        return st.back();
    }

    const std::string& source() const noexcept { return source_; }

  private:
    enum class Kind { Number, VarW, Op1, Op2, Func1, Func2 };
    enum class Op { Add, Sub, Mul, Div, Pow, Neg };
    enum class Fn { Exp, Log, Sqrt, Abs, Sin, Cos, Tan, Step, Pow2, Max, Min };

    struct RpnTok {
        Kind kind{Kind::Number};
        double value{0.0}; // for Number
        Op op{Op::Add};    // for Op1/Op2
        Fn fn{Fn::Exp};    // for Func*
    };

    enum class TokKind { Number, Ident, Op, LParen, RParen, Comma };
    struct Tok {
        TokKind kind{TokKind::Number};
        double number{0.0};
        std::string ident;
        char op{0};
    };

    struct StackItem {
        enum class Type { Op, Func, LParen };
        Type type{Type::Op};
        Op op{Op::Add};
        Fn fn{Fn::Exp};
        int arity{0}; // for Func
    };

    static constexpr double kPi = 3.141592653589793238462643383279502884;
    static constexpr double kE = 2.718281828459045235360287471352662498;

    static std::string lower_copy(std::string s) {
        std::transform(s.begin(), s.end(), s.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return s;
    }

    static bool is_ident_start(char c) {
        return std::isalpha(static_cast<unsigned char>(c)) || c == '_';
    }
    static bool is_ident_char(char c) {
        return std::isalnum(static_cast<unsigned char>(c)) || c == '_';
    }

    static std::vector<Tok> tokenize(std::string_view s) {
        std::vector<Tok> out;
        std::size_t i = 0;
        while (i < s.size()) {
            const char c = s[i];
            if (std::isspace(static_cast<unsigned char>(c))) {
                ++i;
                continue;
            }
            if (c == '(') {
                out.push_back(Tok{TokKind::LParen});
                ++i;
                continue;
            }
            if (c == ')') {
                out.push_back(Tok{TokKind::RParen});
                ++i;
                continue;
            }
            if (c == ',') {
                out.push_back(Tok{TokKind::Comma});
                ++i;
                continue;
            }
            if (c == '+' || c == '-' || c == '*' || c == '/' || c == '^') {
                Tok t;
                t.kind = TokKind::Op;
                t.op = c;
                out.push_back(std::move(t));
                ++i;
                continue;
            }
            if (std::isdigit(static_cast<unsigned char>(c)) || c == '.') {
                const char* begin = s.data() + i;
                char* end = nullptr;
                const double v = std::strtod(begin, &end);
                if (end == begin) {
                    throw std::runtime_error("Invalid number in expression");
                }
                Tok t;
                t.kind = TokKind::Number;
                t.number = v;
                out.push_back(std::move(t));
                i = static_cast<std::size_t>(end - s.data());
                continue;
            }
            if (is_ident_start(c)) {
                std::size_t j = i + 1;
                while (j < s.size() && is_ident_char(s[j])) ++j;
                Tok t;
                t.kind = TokKind::Ident;
                t.ident = std::string(s.substr(i, j - i));
                out.push_back(std::move(t));
                i = j;
                continue;
            }
            throw std::runtime_error(std::string("Unexpected character in expression: '") + c + "'");
        }
        return out;
    }

    static Fn parse_func(const std::string& ident_raw) {
        const std::string ident = lower_copy(ident_raw);
        if (ident == "exp") return Fn::Exp;
        if (ident == "log") return Fn::Log;
        if (ident == "sqrt") return Fn::Sqrt;
        if (ident == "abs") return Fn::Abs;
        if (ident == "sin") return Fn::Sin;
        if (ident == "cos") return Fn::Cos;
        if (ident == "tan") return Fn::Tan;
        if (ident == "step" || ident == "heaviside") return Fn::Step;
        if (ident == "pow") return Fn::Pow2;
        if (ident == "max") return Fn::Max;
        if (ident == "min") return Fn::Min;
        throw std::runtime_error("Unknown function: " + ident_raw);
    }

    static int precedence(Op op) {
        switch (op) {
            case Op::Add:
            case Op::Sub:
                return 1;
            case Op::Mul:
            case Op::Div:
                return 2;
            case Op::Neg:
                return 3;
            case Op::Pow:
                return 4;
        }
        return 0;
    }

    static bool right_assoc(Op op) {
        return (op == Op::Pow || op == Op::Neg);
    }

    static Op parse_op(char c, bool unary_minus) {
        if (unary_minus) return Op::Neg;
        switch (c) {
            case '+': return Op::Add;
            case '-': return Op::Sub;
            case '*': return Op::Mul;
            case '/': return Op::Div;
            case '^': return Op::Pow;
            default: throw std::runtime_error("Unknown operator");
        }
    }

    static double lookup_param(const std::vector<std::pair<std::string, double>>& params,
                               const std::string& name) {
        for (const auto& kv : params) {
            if (kv.first == name) return kv.second;
        }
        throw std::runtime_error("Unknown identifier (not w/pi/e and not in params): " + name);
    }

    static std::vector<RpnTok> compile_to_rpn(const std::string& source,
                                              const std::vector<std::pair<std::string, double>>& params) {
        const auto toks = tokenize(source);
        std::vector<RpnTok> out;
        std::vector<StackItem> st;
        out.reserve(toks.size());
        st.reserve(toks.size());

        auto prev_kind = TokKind::Op; // treat start as if after an operator (so leading '-' is unary)

        for (std::size_t i = 0; i < toks.size(); ++i) {
            const auto& t = toks[i];
            switch (t.kind) {
                case TokKind::Number: {
                    out.push_back(RpnTok{Kind::Number, t.number});
                    prev_kind = TokKind::Number;
                    break;
                }
                case TokKind::Ident: {
                    const bool is_func = (i + 1 < toks.size() && toks[i + 1].kind == TokKind::LParen);
                    if (is_func) {
                        StackItem si;
                        si.type = StackItem::Type::Func;
                        si.fn = parse_func(t.ident);
                        si.arity = 1;
                        st.push_back(std::move(si));
                        prev_kind = TokKind::Ident;
                        break;
                    }

                    const std::string ident = lower_copy(t.ident);
                    if (ident == "w") {
                        out.push_back(RpnTok{Kind::VarW});
                    } else if (ident == "pi") {
                        out.push_back(RpnTok{Kind::Number, kPi});
                    } else if (ident == "e") {
                        out.push_back(RpnTok{Kind::Number, kE});
                    } else {
                        out.push_back(RpnTok{Kind::Number, lookup_param(params, t.ident)});
                    }
                    prev_kind = TokKind::Ident;
                    break;
                }
                case TokKind::LParen: {
                    st.push_back(StackItem{StackItem::Type::LParen});
                    prev_kind = TokKind::LParen;
                    break;
                }
                case TokKind::Comma: {
                    while (!st.empty() && st.back().type != StackItem::Type::LParen) {
                        pop_stack_item_to_output(st.back(), out);
                        st.pop_back();
                    }
                    if (st.empty() || st.back().type != StackItem::Type::LParen) {
                        throw std::runtime_error("Comma outside function call");
                    }
                    if (st.size() < 2 || st[st.size() - 2].type != StackItem::Type::Func) {
                        throw std::runtime_error("Comma outside function call");
                    }
                    st[st.size() - 2].arity += 1;
                    prev_kind = TokKind::Comma;
                    break;
                }
                case TokKind::RParen: {
                    while (!st.empty() && st.back().type != StackItem::Type::LParen) {
                        pop_stack_item_to_output(st.back(), out);
                        st.pop_back();
                    }
                    if (st.empty()) throw std::runtime_error("Mismatched ')'");
                    st.pop_back(); // pop '('

                    if (!st.empty() && st.back().type == StackItem::Type::Func) {
                        const StackItem fn = st.back();
                        st.pop_back();
                        emit_func(fn.fn, fn.arity, out);
                    }
                    prev_kind = TokKind::RParen;
                    break;
                }
                case TokKind::Op: {
                    const bool unary_minus =
                        (t.op == '-') &&
                        (prev_kind == TokKind::Op || prev_kind == TokKind::LParen || prev_kind == TokKind::Comma);
                    const Op op = parse_op(t.op, unary_minus);
                    const int prec = precedence(op);
                    const bool right = right_assoc(op);

                    while (!st.empty() && st.back().type == StackItem::Type::Op) {
                        const Op top = st.back().op;
                        const int ptop = precedence(top);
                        if (ptop > prec || (!right && ptop == prec)) {
                            pop_stack_item_to_output(st.back(), out);
                            st.pop_back();
                        } else {
                            break;
                        }
                    }
                    StackItem si;
                    si.type = StackItem::Type::Op;
                    si.op = op;
                    st.push_back(std::move(si));
                    prev_kind = TokKind::Op;
                    break;
                }
            }
        }

        while (!st.empty()) {
            if (st.back().type == StackItem::Type::LParen) throw std::runtime_error("Mismatched '('");
            pop_stack_item_to_output(st.back(), out);
            st.pop_back();
        }

        return out;
    }

    static void emit_func(Fn fn, int arity, std::vector<RpnTok>& out) {
        if (arity == 1) out.push_back(RpnTok{Kind::Func1, 0.0, Op::Add, fn});
        else if (arity == 2) out.push_back(RpnTok{Kind::Func2, 0.0, Op::Add, fn});
        else throw std::runtime_error("Unsupported function arity (only 1 or 2 args supported)");
    }

    static void pop_stack_item_to_output(const StackItem& si, std::vector<RpnTok>& out) {
        if (si.type == StackItem::Type::Op) {
            if (si.op == Op::Neg) out.push_back(RpnTok{Kind::Op1, 0.0, Op::Neg});
            else out.push_back(RpnTok{Kind::Op2, 0.0, si.op});
            return;
        }
        if (si.type == StackItem::Type::Func) {
            emit_func(si.fn, si.arity, out);
            return;
        }
        throw std::runtime_error("Internal error: unexpected stack item");
    }

    std::string source_;
    std::vector<RpnTok> rpn_;
};

} // namespace taco::expr

