import base from './base.js';

/** @type {import('eslint').Linter.FlatConfig[]} */
export default [
  ...base,
  {
    rules: {
      'no-console': 'off',
    },
  },
];
