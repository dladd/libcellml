/*
Copyright 2015 University of Auckland

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include "gtest/gtest.h"

#include <libcellml>

// SymEngine declarations
#include <symengine/mathml_printer.h>
using SymEngine::RCP;
using SymEngine::Basic;
using SymEngine::div;
using SymEngine::Expression;
using SymEngine::pow;
using SymEngine::UIntPoly;
using SymEngine::uexpr_poly;
using SymEngine::mul;
using SymEngine::integer;
using SymEngine::print_stack_on_segfault;
using SymEngine::symbol;
using SymEngine::Complex;
using SymEngine::Rational;
using SymEngine::Number;
using SymEngine::add;
using SymEngine::Symbol;
using SymEngine::erf;
using SymEngine::Integer;
using SymEngine::loggamma;
using SymEngine::Subs;
using SymEngine::Derivative;
using SymEngine::function_symbol;
using SymEngine::I;
using SymEngine::real_double;
using SymEngine::complex_double;
using SymEngine::BaseVisitor;
using SymEngine::StrPrinter;
using SymEngine::MathMLPrinter;
using SymEngine::Sin;
using SymEngine::integer_class;
using SymEngine::map_uint_mpz;
using SymEngine::Infty;
using SymEngine::infty;
using SymEngine::E;
using namespace SymEngine::literals;

TEST(Maths, algebraicEquation) {
    RCP<const Basic> eq1 = symbol("eq1");
    RCP<const Symbol> a = symbol("a");
    RCP<const Symbol> b = symbol("b");
    std::string expected = "-1 + a + b";

    eq1 = sub(add(a, b), integer(1));
    EXPECT_EQ(expected, eq1->__str__());
}

TEST(Maths, algebraicFunction) {
    RCP<const Basic> eq1 = symbol("eq1");
    RCP<const Symbol> t = symbol("t");
    RCP<const Symbol> a = symbol("a");
    RCP<const Symbol> b = symbol("b");
    RCP<const Symbol> y0 = symbol("y0");
    RCP<const Basic> y = function_symbol("y",t);
    std::string expected = "b/a + (y0 - b/a)*E**(-a*t) - y(t)";

    eq1 = sub(add(div(b, a), mul(sub(y0, div(b, a)),
          pow(E, mul(integer(-1), mul(a, t))))), y);
    EXPECT_EQ(expected, eq1->__str__());
}

TEST(Maths, differentialEquation) {
    RCP<const Basic> eq1 = symbol("eq1");
    RCP<const Symbol> t = symbol("t");
    RCP<const Symbol> a = symbol("a");
    RCP<const Symbol> b = symbol("b");
    RCP<const Basic> y = function_symbol("y",t);
    std::string expected = "-b + a*y(t) + Derivative(y(t), t)";

    eq1 = sub(add(y->diff(t), mul(a,y)), b);
    EXPECT_EQ(expected, eq1->__str__());
}

TEST(Maths, mathmlSin) {
    SymEngine::MathMLPrinter printer;
    RCP<const Basic> p = symbol("p");
    RCP<const Symbol> x = symbol("x");
    std::string expected =
            "<apply>"
                "<sin/>"
                "<ci>x</ci>"
            "</apply>";

    p = sin(x);
    std::string math = printer.apply(p);
    EXPECT_EQ(math, expected);
}

TEST(Maths, mathmlCosSin) {
    SymEngine::MathMLPrinter printer;
    RCP<const Basic> p = symbol("p");
    RCP<const Symbol> x = symbol("x");
    std::string expected =
            "<apply>"
                "<cos/>"
                "<apply>"
                    "<sin/>"
                    "<ci>x</ci>"
                "</apply>"
            "</apply>";

    p = cos(sin(x));
    std::string math = printer.apply(p);
    EXPECT_EQ(math, expected);
}

TEST(Maths, mathmlXPlusOne) {
    SymEngine::MathMLPrinter printer;
    RCP<const Basic> p = symbol("p");
    RCP<const Symbol> x = symbol("x");
    RCP<const Integer> one = integer(1);
    std::string expected =
            "<apply>"
                "<plus/>"
                "<cn>1</cn>"
                "<ci>x</ci>"
            "</apply>";

    p = add(x, one);
    std::string math = printer.apply(p);
    EXPECT_EQ(math, expected);
}


