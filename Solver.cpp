#include <algorithm>
#include <cassert>
#include <cstddef>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <thread>
#include <vector>

#include <gflags/gflags.h>
using namespace std;

DEFINE_bool(has_answer, false, "When true, Solver will assume that exact answer is provided");
DEFINE_int32(num_threads, 8, "Number of threads");

enum class Where { Up, Down, Left, Right };
const int kNumDirs = 4;
const int kDRow[] = {-1, +1, 0, 0};
const int kDCol[] = {0, 0, -1, +1};
const Where kDirs[] = {Where::Up, Where::Down, Where::Left, Where::Right};
const int kFreeCell = -1;

mt19937 g_engine{42};

template <typename T>
ostream& operator<<(ostream& os, const vector<T>& vs) {
  bool first = true;
  for (const auto& v : vs) {
    if (!first)
      os << " ";
    os << v;
    first = false;
  }
  return os;
}

struct Context {
  vector<vector<double>> m_hpreds;
  vector<vector<double>> m_vpreds;
  int m_size = 0;
  int m_total = 0;
};

istream& operator>>(istream& is, Context& ctx) {
  is >> ctx.m_size;
  ctx.m_total = ctx.m_size * ctx.m_size;

  ctx.m_hpreds.assign(ctx.m_total, vector<double>(ctx.m_total));
  for (int i = 0; i < ctx.m_total; ++i) {
    for (int j = 0; j < ctx.m_total; ++j)
      is >> ctx.m_hpreds[i][j];
  }
  ctx.m_vpreds.assign(ctx.m_total, vector<double>(ctx.m_total));
  for (int i = 0; i < ctx.m_total; ++i) {
    for (int j = 0; j < ctx.m_total; ++j)
      is >> ctx.m_vpreds[i][j];
  }

  return is;
}

template <typename T>
void SortUnique(vector<T>& vs) {
  sort(vs.begin(), vs.end());
  vs.erase(unique(vs.begin(), vs.end()), vs.end());
}

template <typename T>
void RemoveAll(vector<T>& vs, const T& v) {
  auto it = remove(vs.begin(), vs.end(), v);
  vs.erase(it, vs.end());
}

template <typename T>
size_t SetIntersectionSize(const vector<T>& lhs, const vector<T>& rhs) {
  size_t i = 0, j = 0;
  size_t size = 0;
  while (i < lhs.size() && j < rhs.size()) {
    if (lhs[i] < rhs[j]) {
      ++i;
    } else if (lhs[i] > rhs[j]) {
      ++j;
    } else {
      ++size;
      ++i;
      ++j;
    }
  }
  return size;
}

double GetScore(int size, const vector<int>& expected, const vector<int>& actual) {
  using Pair = pair<int, int>;

  vector<Pair> hexp;
  vector<Pair> hact;
  for (int r = 0; r < size; ++r) {
    for (int c = 0; c + 1 < size; ++c) {
      hexp.emplace_back(expected[r * size + c], expected[r * size + c + 1]);
      hact.emplace_back(actual[r * size + c], actual[r * size + c + 1]);
    }
  }

  vector<Pair> vexp;
  vector<Pair> vact;
  for (int r = 0; r + 1 < size; ++r) {
    for (int c = 0; c < size; ++c) {
      vexp.emplace_back(expected[r * size + c], expected[(r + 1) * size + c]);
      vact.emplace_back(actual[r * size + c], actual[(r + 1) * size + c]);
    }
  }

  const double a = SetIntersectionSize(hexp, hact) + SetIntersectionSize(vexp, vact);
  const double b = 2 * size * (size - 1);
  return a / b;
}

struct Piece {
  Piece() = default;
  Piece(int row, int col, int piece) : m_row(row), m_col(col), m_piece(piece) {}

  int m_row = 0;
  int m_col = 0;
  int m_piece = 0;
};

struct Cell {
  Cell() = default;
  Cell(int row, int col) : m_row(row), m_col(col) {}

  int m_row = 0;
  int m_col = 0;
};

struct BoundingBox {
  BoundingBox() = default;
  BoundingBox(int minRow, int minCol, int maxRow, int maxCol)
      : m_minRow(minRow), m_minCol(minCol), m_maxRow(maxRow), m_maxCol(maxCol) {}

  int NumRows() const {
    assert(m_minRow <= m_maxRow);
    return m_maxRow - m_minRow;
  }

  int NumCols() const {
    assert(m_minCol <= m_maxCol);
    return m_maxCol - m_minCol;
  }

  void AddInplace(const Cell& cell) {
    m_minRow = min(m_minRow, cell.m_row);
    m_minCol = min(m_minCol, cell.m_col);
    m_maxRow = max(m_maxRow, cell.m_row + 1);
    m_maxCol = max(m_maxCol, cell.m_col + 1);
  }

  [[nodiscard]] BoundingBox Add(const Cell& cell) const {
    auto bb = *this;
    bb.AddInplace(cell);
    return bb;
  }

  int m_minRow = numeric_limits<int>::max();
  int m_minCol = numeric_limits<int>::max();
  int m_maxRow = numeric_limits<int>::min();
  int m_maxCol = numeric_limits<int>::min();
};

bool IsValid(const vector<vector<int>>& field, int r, int c) {
  if (r < 0 || r >= field.size())
    return false;
  if (c < 0 || c >= field[r].size())
    return false;
  return true;
}

double GetPredict(int p, int q, Where where, const Context& ctx) {
  switch (where) {
    case Where::Left:
      return ctx.m_hpreds[p][q];
    case Where::Right:
      return ctx.m_hpreds[q][p];
    case Where::Up:
      return ctx.m_vpreds[p][q];
    case Where::Down:
      return ctx.m_vpreds[q][p];
  }
}

template <typename TFn>
void ForEachCandidateCell(const vector<vector<int>>& field, const BoundingBox& bbox, const Context& ctx, TFn&& fn) {
  assert(bbox.NumRows() <= ctx.m_size);
  assert(bbox.NumCols() <= ctx.m_size);

  for (int row = 0; row < field.size(); ++row) {
    for (int col = 0; col < field[0].size(); ++col) {
      if (field[row][col] != kFreeCell)
        continue;
      const auto bb = bbox.Add(Cell(row, col));
      if (bb.NumRows() > ctx.m_size || bb.NumCols() > ctx.m_size)
        continue;

      bool good = false;
      for (int k = 0; k < kNumDirs && !good; ++k) {
        const auto r = row + kDRow[k];
        const auto c = col + kDCol[k];

        if (IsValid(field, r, c))
          good = good || field[r][c] != kFreeCell;
      }
      if (good)
        fn(Cell(row, col));
    }
  }
}

struct Solution {
  Solution() = default;

  template <typename P>
  Solution(P&& permutation, double score) : m_permutation(std::forward<P>(permutation)), m_score(score) {}

  vector<int> m_permutation;
  double m_score = 0;
};

Solution HillClimbing(const Context& ctx) {
  if (ctx.m_total == 0)
    return {};

  vector<vector<int>> field(2 * ctx.m_size, vector<int>(2 * ctx.m_size, kFreeCell));
  BoundingBox bbox;
  field[ctx.m_size][ctx.m_size] = 0;

  vector<int> avail(ctx.m_total);
  iota(avail.begin(), avail.end(), 0);

  shuffle(avail.begin(), avail.end(), g_engine);

  auto markUsed = [&](const Cell& cell, int p) {
    field[cell.m_row][cell.m_col] = p;
    bbox.AddInplace(cell);
    RemoveAll(avail, p);
  };

  markUsed(Cell(ctx.m_size, ctx.m_size), avail[0]);
  double totalScore = 0;

  for (int i = 1; i < ctx.m_total; ++i) {
    Cell bestCell;
    double bestScore = -1;
    int bestP = -1;

    vector<Cell> cells;
    ForEachCandidateCell(field, bbox, ctx, [&](const Cell& cell) { cells.push_back(cell); });
    shuffle(cells.begin(), cells.end(), g_engine);

    assert(!cells.empty());
    assert(!avail.empty());
    for (const auto& cell : cells) {
      for (const auto& p : avail) {
        assert(field[cell.m_row][cell.m_col] == kFreeCell);
        double score = 0;
        int matched = 0;
        int total = 0;
        for (int k = 0; k < kNumDirs; ++k) {
          const auto r = cell.m_row + kDRow[k];
          const auto c = cell.m_col + kDCol[k];
          if (IsValid(field, r, c) && field[r][c] != kFreeCell) {
            const auto q = field[r][c];
            const auto predict = GetPredict(q, p, kDirs[k], ctx);
            score += predict;
            if (predict > 0.5)
              ++matched;
            ++total;
          }
        }

        if (bestScore < score * matched / total) {
          bestScore = score * matched / total;
          bestCell = cell;
          bestP = p;
        }
      }
    }
    assert(bestScore >= 0);
    assert(bestP >= 0);

    markUsed(bestCell, bestP);

    totalScore += bestScore;
  }

  assert(avail.empty());
  assert(bbox.NumRows() == ctx.m_size);
  assert(bbox.NumCols() == ctx.m_size);

  vector<int> result;
  for (int r = bbox.m_minRow; r < bbox.m_maxRow; ++r) {
    for (int c = bbox.m_minCol; c < bbox.m_maxCol; ++c) {
      result.push_back(field[r][c]);
    }
  }

  const auto totalEdges = 2 * ctx.m_size * (ctx.m_size - 1);
  return Solution{std::move(result), totalScore / totalEdges};
}

bool CheckSolution(const Context& ctx, const vector<int>& actual) {
  vector<bool> used(ctx.m_total);

  if (actual.size() != ctx.m_total) {
    cerr << "Wrong result size: expected " << ctx.m_total << ", actual: " << actual.size()
         << " for result permutation: " << actual;
    return false;
  }

  for (const auto& p : actual) {
    if (p < 0 || p >= ctx.m_total) {
      cerr << "Incorrect value " << p << " in result permutation: " << actual;
      return false;
    }
    if (used[p]) {
      cerr << "Duplicate value " << p << " in result permutation: " << actual;
      return false;
    }
    used[p] = true;
  }

  return true;
}

vector<int> Solve(const Context& ctx) {
  Solution best = HillClimbing(ctx);
  for (int i = 0; i < 10000; ++i) {
    const auto curr = HillClimbing(ctx);
    if (curr.m_score > best.m_score)
      best = curr;
  }
  cerr << "Best solution score: " << best.m_score << endl;
  return best.m_permutation;
}

vector<int> Solve(istream& is, const string& tag) {
  Context ctx;
  is >> ctx;

  vector<int> expected;
  if (FLAGS_has_answer) {
    expected.assign(ctx.m_total, 0);
    for (int i = 0; i < ctx.m_total; ++i)
      is >> expected[i];
  }

  const auto actual = Solve(ctx);
  if (!CheckSolution(ctx, actual)) {
    cerr << tag << ": solution check failed, exiting..." << endl;
    exit(EXIT_FAILURE);
  }

  if (FLAGS_has_answer)
    cerr << tag << ": result score: " << GetScore(ctx.m_size, expected, actual) << endl;

  return actual;
}

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, /* remove_flags= */ true);
  ios_base::sync_with_stdio(false);

  vector<string> inputs;
  for (auto** curr = argv + 1; *curr; ++curr)
    inputs.push_back(*curr);

  vector<vector<int>> outputs(inputs.size());

  auto solver = [&](int offset) {
    for (int i = offset; i < inputs.size(); i += FLAGS_num_threads) {
      ifstream is(inputs[i]);
      outputs[i] = Solve(is, inputs[i]);
    }
  };

  vector<thread> threads;
  for (int i = 0; i < FLAGS_num_threads; ++i)
    threads.emplace_back(bind(solver, i));

  for (auto& thread : threads)
    thread.join();

  for (int i = 0; i < inputs.size(); ++i) {
    cout << inputs[i] << endl;
    cout << outputs[i] << endl;
  }

  exit(EXIT_SUCCESS);
}
