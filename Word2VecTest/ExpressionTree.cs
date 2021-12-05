using Accord.Math;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.WordEmbeddings
{
    public class Node
    {
        public Node Left;
        public Node Right;
        public double[] Data;
        public string operation;
    }
    public class ExpressionTree
    {
        // Evaluate a string-based, postfix expression
        public static double[] EvaluateExpression(object[] terms)
        {
            // Create tree
            var nodes = CreateExpressionTree(terms);

            // Traverse tree starting from root
            var root = nodes.Last();
            Traverse(root);

            // Return final expression result
            return root.Data;
        }

        // Create a new binary expression tree
        // based on a postfix expression
        private static Node[] CreateExpressionTree(object[] terms)
        {
            // Init tree
            var tree = new Node[terms.Length];

            // Map terms onto tree
            var nodes = terms.Select(x => {
                if (x is double[])
                {
                    return new Node { Data = x as double[] };
                }
                else
                {
                    return new Node { operation = x as string };
                }
            }).ToArray();

            // Create tree
            for (int i = 0; i < terms.Length; i++)
            {
                var current = nodes[i];

                // Check if current node is an operator
                if (current.operation != null)
                {
                    current.Left = nodes[i - 2];
                    current.Right = nodes[i - 1];
                }
            }

            return nodes;
        }

        // Traverse tree and evaluate nodes
        private static void Traverse(Node node)
        {
            if (node.Left != null)
            {
                Traverse(node.Left);
            }
            if (node.Right != null)
            {
                Traverse(node.Right);
            }

            // Implement expression
            // if current node is an operator
            if (node.operation == "+")
            {
                node.Data = node.Left.Data.Add(node.Right.Data);
            }
            else if (node.operation == "-")
            {
                node.Data = node.Left.Data.Subtract(node.Right.Data);
            }
        }
    }
}
