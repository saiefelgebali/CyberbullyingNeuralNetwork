using Accord.Math;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace NeuralNetwork.WordEmbeddings
{
    internal class PostfixExpression
    {
        // Regex patterns
        private const string CharacterPattern = "[A-Za-z0-9]";
        private const string WordPattern = $"{CharacterPattern}+";
        private const string OperationPattern = "[+|-]";
        private const string ValidInfixExpression = $"({WordPattern}{OperationPattern})+{WordPattern}";

        // Properties
        public string[] InfixTerms { get; private set; }
        public string[] PostfixTerms { get; private set; }
        public object[] PostfixTermsParsed { get; private set; }

        public PostfixExpression(string expression, Func<string, object> convertString)
        {
            // Remove spaces
            expression = Regex.Replace(expression, "[ ]*", "");

            // Check validity
            if (!IsValidInfix(expression))
            {
                throw new ArgumentException("String is not a valid infix expression");
            }

            // Convert Expression into array of terms
            InfixTerms = ParseExpression(expression);
            PostfixTerms = InfixToPostfix(InfixTerms);

            // Convert string to numerical value
            PostfixTermsParsed = PostfixTerms.Select(x =>
            {
                //  Convert non-operators into their numerical values
                if (!Regex.IsMatch(x, OperationPattern)) return convertString(x);
                return x;
            }).ToArray();
        }

        static bool IsValidInfix(string expression)
        {
            // Check if entire expression is valid for infix
            var match = Regex.Match(expression, ValidInfixExpression);

            return match.Length == expression.Length;
        }

        static string[] ParseExpression(string expression)
        {
            // Convert string expression
            // into an array of string terms
            return Regex.Matches(expression, $"({WordPattern}|{OperationPattern})")
                .Select(t => t.ToString()).ToArray();
        }

        static string[] InfixToPostfix(string[] infix)
        {
            var postfix = infix.Copy();

            // Shift all operators by one space
            // to the right to create postfix expression
            for (int i = 0; i < infix.Length; i++)
            {
                // If a valid operator
                if (Regex.IsMatch(infix[i], OperationPattern))
                {
                    // Switch terms
                    var current = infix[i];
                    var nextTerm = infix[i + 1];
                    postfix[i + 1] = current;
                    postfix[i] = nextTerm;
                }
            }

            // Return updated array
            return postfix;
        }
    }
}
