# Crossing Audit Report: celery

**Project:** celery (celery/celery)
**Scanned:** 2026-02-24
**Tool:** Crossing Semantic Scanner v0.9

---

## Executive Summary

celery has **40 semantic boundary crossings**, including **4 high-risk** findings. For a 161-file codebase with 292 raise sites and 554 handlers, this gives a crossing density of 0.25 per file.

**Risk Level:** High.

---

## Scan Summary

| Metric | Value |
|--------|-------|
| Files scanned | 161 |
| Raise sites | 292 |
| Exception handlers | 554 |
| Total crossings | 40 |
| High risk | 4 |
| Elevated risk | 0 |
| Medium risk | 8 |
| Low risk | 28 |
| Mean collapse ratio | 46% |

---

## Findings

### HIGH RISK: `KeyError` — 19 raise sites, 112 handlers

**Files:** `__init__.py`, `app/amqp.py`, `app/annotations.py`, `app/defaults.py`, `app/registry.py`, `app/routes.py`, `app/trace.py`, `app/utils.py`, `apps/multi.py`, `backends/asynchronous.py`, `backends/base.py`, `backends/cache.py`, `backends/database/session.py`, `backends/redis.py`, `backends/rpc.py`, `bin/base.py`, `bin/graph.py`, `bin/multi.py`, `bootsteps.py`, `canvas.py`, `concurrency/__init__.py`, `concurrency/asynpool.py`, `concurrency/gevent.py`, `events/cursesmon.py`, `events/dumper.py`, `events/receiver.py`, `events/state.py`, `loaders/base.py`, `platforms.py`, `result.py`, `schedules.py`, `security/certificate.py`, `states.py`, `utils/collections.py`, `utils/graph.py`, `utils/imports.py`, `utils/serialization.py`, `utils/text.py`, `utils/threads.py`, `utils/time.py`, `worker/consumer/consumer.py`, `worker/consumer/gossip.py`, `worker/control.py`, `worker/pidbox.py`, `worker/state.py`, `worker/strategy.py`, `worker/worker.py`
**Impact:** `KeyError` is raised at 19 sites across 10 files (__init__.py, amqp.py, asynchronous.py, bootsteps.py, collections.py, multi.py, platforms.py, result.py, routes.py, time.py), in 18 different functions. 112 handlers (22 re-raise, 26 return, 18 assign default). Information collapse: 80% of semantic information is lost (3.4 bits destroyed).

**Raise sites:**
- `bootsteps.py:249` raise `KeyError` in `Blueprint._finalize_steps` — in _finalize_steps
- `result.py:596` raise `KeyError` in `ResultSet.remove` — in remove
- `platforms.py:446` raise `KeyError` in `parse_uid` — in parse_uid
- `platforms.py:463` raise `KeyError` in `parse_gid` — in parse_gid
- `__init__.py:104` raise `KeyError` in `_find_option_with_arg` — if short_opts and arg in short_opts → raise in _find_option_with_arg
- `app/amqp.py:100` raise `KeyError` in `Queues.__missing__` — if self.create_missing → raise in __missing__
- `apps/multi.py:241` raise `KeyError` in `Node.getopt` — in getopt
- `apps/multi.py:341` raise `KeyError` in `MultiParser._update_ns_opts` — if ns_index < 0 → raise in _update_ns_opts
- ... and 11 more

**Handlers:**
- `schedules.py:306` — except `KeyError` in `crontab_parser._expand_number` (handles)
- `schedules.py:309` — except `KeyError` in `crontab_parser._expand_number` (re-raises)
- `bootsteps.py:248` — except `KeyError` in `Blueprint._finalize_steps` (re-raises)
- `canvas.py:501` — except `KeyError` in `Signature.freeze` (assigns default)
- `canvas.py:852` — except `KeyError` in `Signature.AsyncResult` (returns)
- ... and 107 more

**Information theory:** 4.2 bits entropy, 3.4 bits lost, 80% collapse

**Recommendation:** `KeyError` is a broad built-in type carrying 19 different meanings here. Consider defining project-specific exception subclasses, or narrowing handler try-blocks to minimize the catch surface.

### HIGH RISK: `Exception` — 92 raise sites, 85 handlers

**Files:** `_state.py`, `app/backends.py`, `app/base.py`, `app/builtins.py`, `app/control.py`, `app/registry.py`, `app/task.py`, `app/trace.py`, `app/utils.py`, `apps/beat.py`, `backends/arangodb.py`, `backends/asynchronous.py`, `backends/azureblockblob.py`, `backends/base.py`, `backends/cache.py`, `backends/cassandra.py`, `backends/consul.py`, `backends/cosmosdbsql.py`, `backends/couchbase.py`, `backends/couchdb.py`, `backends/database/__init__.py`, `backends/dynamodb.py`, `backends/elasticsearch.py`, `backends/filesystem.py`, `backends/gcs.py`, `backends/mongodb.py`, `backends/redis.py`, `backends/s3.py`, `beat.py`, `bin/amqp.py`, `bin/celery.py`, `bin/worker.py`, `bootsteps.py`, `concurrency/asynpool.py`, `concurrency/base.py`, `concurrency/prefork.py`, `contrib/migrate.py`, `contrib/rdb.py`, `contrib/testing/manager.py`, `events/cursesmon.py`, `events/dispatcher.py`, `events/receiver.py`, `fixups/django.py`, `loaders/base.py`, `local.py`, `platforms.py`, `result.py`, `security/__init__.py`, `security/certificate.py`, `utils/dispatch/signal.py`, `utils/imports.py`, `utils/log.py`, `utils/saferepr.py`, `utils/serialization.py`, `utils/threads.py`, `utils/time.py`, `utils/timer2.py`, `worker/autoscale.py`, `worker/components.py`, `worker/consumer/consumer.py`, `worker/consumer/delayed_delivery.py`, `worker/consumer/gossip.py`, `worker/consumer/mingle.py`, `worker/control.py`, `worker/loops.py`, `worker/pidbox.py`, `worker/request.py`, `worker/strategy.py`, `worker/worker.py`
**Impact:** `Exception` is raised at 92 sites across 38 files (__init__.py, _state.py, arangodb.py, asynchronous.py, azureblockblob.py, backends.py, base.py, cache.py, cassandra.py, certificate.py, components.py, consul.py, control.py, cosmosdbsql.py, couchbase.py, couchdb.py, django.py, dynamodb.py, elasticsearch.py, filesystem.py, gcs.py, imports.py, manager.py, migrate.py, mongodb.py, platforms.py, rdb.py, receiver.py, redis.py, registry.py, request.py, result.py, s3.py, strategy.py, task.py, trace.py, utils.py, worker.py), in 66 different functions. 85 handlers (10 re-raise, 12 return, 20 assign default). Information collapse: 88% of semantic information is lost (5.3 bits destroyed).

**Raise sites:**
- `_state.py:175` raise `Exception` in `_app_or_default_trace` — if not current_process or current_process()._name == 'MainProcess' → raise in _app_or_default_trace (`"DEFAULT APP`)
- `contrib/rdb.py:135` raise `Exception` in `Rdb.get_avail_port` — if exc.errno in [errno.EADDRINUSE, errno.EINVAL] → raise in get_avail_port
- `backends/asynchronous.py:163` raise `Exception` in `greenletDrainer._ensure_not_shut_down` — if self._exc is not None → raise in _ensure_not_shut_down
- `result.py:767` raise `Exception` in `ResultSet.join` — if on_message is not None → raise in join (`"Backend does not support on_message callback`)
- `app/base.py:117` raise `Exception` in `pydantic_wrapper` — in pydantic_wrapper (`"You need to install pydantic to use pydantic model serializa...`)
- `app/base.py:715` raise `Exception` in `Celery.config_from_envvar` — if silent → raise in config_from_envvar
- `app/utils.py:274` raise `Exception` in `detect_settings` — if really_left → raise in detect_settings
- `app/task.py:968` raise `Exception` in `Task.replace` — if 'chord' in sig.options → raise in replace (`"A signature replacing a task must not be part of a chord`)
- ... and 84 more

**Handlers:**
- `bootsteps.py:149` — except `Exception` in `Blueprint.send_all` (handles)
- `beat.py:283` — except `Exception` in `Scheduler.apply_entry` (handles)
- `beat.py:411` — except `Exception` in `Scheduler.apply_async` (handles)
- `beat.py:540` — except `Exception` in `PersistentScheduler.setup_schedule` (assigns default)
- `platforms.py:490` — except `Exception` in `setgroups` (handles)
- ... and 80 more

**Information theory:** 6.0 bits entropy, 5.3 bits lost, 88% collapse

**Recommendation:** Multiple handlers exist, which may provide adequate discrimination. Verify that each handler's try-block scope only exposes the expected raise sites.

### HIGH RISK: `SystemExit` — 3 raise sites, 1 handler

**Files:** `apps/beat.py`, `events/snapshot.py`, `platforms.py`, `worker/worker.py`
**Impact:** `SystemExit` is raised at 3 sites across 3 files (beat.py, platforms.py, snapshot.py), in 3 different functions. A single handler in `WorkController.start` handles. With 3 raise sites funneling through one handler, semantic disambiguation is impossible. Information collapse: 100% of semantic information is lost (1.6 bits destroyed).

**Raise sites:**
- `platforms.py:270` raise `SystemExit` in `_create_pidlock` — if pidlock.is_locked() and not pidlock.remove_if_stale() → raise in _create_pidlock
- `apps/beat.py:159` raise `SystemExit` in `Beat._sync` — in _sync
- `events/snapshot.py:108` raise `SystemExit` in `evcam` — in evcam

**Handlers:**
- `worker/worker.py:209` — except `SystemExit` in `WorkController.start` (handles)

**Information theory:** 1.6 bits entropy, 1.6 bits lost, 100% collapse

**Recommendation:** Consider using distinct exception subclasses for the 3 different error conditions, or narrow the handler to catch only from the expected call site.

### HIGH RISK: `WorkerTerminate` — 3 raise sites, 1 handler

**Files:** `apps/worker.py`, `worker/consumer/consumer.py`, `worker/state.py`, `worker/worker.py`
**Impact:** `WorkerTerminate` is raised at 3 sites across 3 files (consumer.py, state.py, worker.py), in 3 different functions. A single handler in `WorkController.start` handles. With 3 raise sites funneling through one handler, semantic disambiguation is impossible. Information collapse: 100% of semantic information is lost (1.6 bits destroyed).

**Raise sites:**
- `apps/worker.py:335` raise `WorkerTerminate` in `on_hard_shutdown` — in on_hard_shutdown
- `worker/state.py:91` raise `WorkerTerminate` in `maybe_shutdown` — if should_terminate is not None and should_terminate is not False → raise in maybe_shutdown
- `worker/consumer/consumer.py:363` raise `WorkerTerminate` in `Consumer.start` — if isinstance(exc, OSError) and exc.errno == errno.EMFILE → raise in start

**Handlers:**
- `worker/worker.py:204` — except `WorkerTerminate` in `WorkController.start` (handles)

**Information theory:** 1.6 bits entropy, 1.6 bits lost, 100% collapse

**Recommendation:** Consider using distinct exception subclasses for the 3 different error conditions, or narrow the handler to catch only from the expected call site.

### MEDIUM RISK: `NotImplementedError` — 29 raise sites, 5 handlers

**Files:** `app/builtins.py`, `app/task.py`, `backends/asynchronous.py`, `backends/base.py`, `backends/rpc.py`, `beat.py`, `bin/list.py`, `bootsteps.py`, `concurrency/base.py`, `schedules.py`, `utils/collections.py`, `utils/threads.py`, `worker/worker.py`
**Impact:** `NotImplementedError` is raised at 29 sites across 10 files (asynchronous.py, base.py, bootsteps.py, builtins.py, collections.py, rpc.py, schedules.py, task.py, threads.py, worker.py), in 29 different functions. 5 handlers (1 re-raise, 3 assign default). Information collapse: 80% of semantic information is lost (3.9 bits destroyed).

**Raise sites:**
- `schedules.py:75` raise `NotImplementedError` in `BaseSchedule.remaining_estimate` — in remaining_estimate
- `schedules.py:78` raise `NotImplementedError` in `BaseSchedule.is_due` — in is_due
- `bootsteps.py:393` raise `NotImplementedError` in `ConsumerStep.get_consumers` — in get_consumers (`"missing get_consumers`)
- `app/builtins.py:161` raise `NotImplementedError` in `chain` — in chain (`"chain is not a real task`)
- `app/task.py:447` raise `NotImplementedError` in `Task.run` — in run (`"Tasks must define the run method.`)
- `backends/asynchronous.py:299` raise `NotImplementedError` in `BaseResultConsumer.start` — in start
- `backends/asynchronous.py:305` raise `NotImplementedError` in `BaseResultConsumer.drain_events` — in drain_events
- `backends/asynchronous.py:308` raise `NotImplementedError` in `BaseResultConsumer.consume_from` — in consume_from
- ... and 21 more

**Handlers:**
- `beat.py:702` — except `NotImplementedError` in `<module>` (assigns default)
- `worker/worker.py:112` — except `NotImplementedError` in `WorkController.setup_instance` (assigns default)
- `worker/worker.py:278` — except `NotImplementedError` in `WorkController.reload` (handles)
- `worker/worker.py:332` — except `NotImplementedError` in `WorkController.stats` (assigns default)
- `bin/list.py:30` — except `NotImplementedError` in `bindings` (re-raises)

**Information theory:** 4.9 bits entropy, 3.9 bits lost, 80% collapse

**Recommendation:** Multiple handlers exist, which may provide adequate discrimination. Verify that each handler's try-block scope only exposes the expected raise sites.

### MEDIUM RISK: `ValueError` — 45 raise sites, 30 handlers

**Files:** `app/amqp.py`, `app/backends.py`, `app/base.py`, `app/routes.py`, `app/task.py`, `apps/multi.py`, `backends/base.py`, `backends/cosmosdbsql.py`, `backends/dynamodb.py`, `backends/gcs.py`, `backends/redis.py`, `bin/base.py`, `bin/multi.py`, `bin/worker.py`, `concurrency/asynpool.py`, `contrib/rdb.py`, `loaders/base.py`, `platforms.py`, `result.py`, `schedules.py`, `security/certificate.py`, `security/key.py`, `utils/collections.py`, `utils/dispatch/signal.py`, `utils/imports.py`, `utils/iso8601.py`, `utils/serialization.py`, `utils/text.py`, `worker/autoscale.py`, `worker/consumer/delayed_delivery.py`, `worker/control.py`
**Impact:** `ValueError` is raised at 45 sites across 19 files (amqp.py, asynpool.py, base.py, certificate.py, collections.py, control.py, cosmosdbsql.py, delayed_delivery.py, gcs.py, iso8601.py, key.py, platforms.py, redis.py, result.py, schedules.py, serialization.py, signal.py, task.py, text.py), in 33 different functions. 30 handlers (5 re-raise, 7 return, 3 assign default). Information collapse: 80% of semantic information is lost (4.0 bits destroyed).

**Raise sites:**
- `schedules.py:310` raise `ValueError` in `crontab_parser._expand_number` — in _expand_number
- `schedules.py:314` raise `ValueError` in `crontab_parser._expand_number` — if i > max_val → raise in _expand_number
- `schedules.py:317` raise `ValueError` in `crontab_parser._expand_number` — if i < self.min_ → raise in _expand_number
- `schedules.py:476` raise `ValueError` in `crontab._expand_cronspec` — if number >= max_ + min_ or number < min_ → raise in _expand_cronspec
- `schedules.py:836` raise `ValueError` in `solar.__init__` — if event not in self._all_events → raise in __init__
- `schedules.py:840` raise `ValueError` in `solar.__init__` — if lat < -90 or lat > 90 → raise in __init__
- `schedules.py:842` raise `ValueError` in `solar.__init__` — if lon < -180 or lon > 180 → raise in __init__
- `result.py:92` raise `ValueError` in `AsyncResult.__init__` — if id is None → raise in __init__
- ... and 37 more

**Handlers:**
- `schedules.py:303` — except `ValueError` in `crontab_parser._expand_number` (handles)
- `schedules.py:498` — except `ValueError` in `crontab.day_out_of_range` (returns)
- `result.py:595` — except `ValueError` in `ResultSet.remove` (re-raises)
- `platforms.py:174` — except `ValueError` in `Pidfile.read_pid` (re-raises)
- `platforms.py:190` — except `ValueError` in `Pidfile.remove_if_stale` (returns)
- ... and 25 more

**Information theory:** 5.0 bits entropy, 4.0 bits lost, 80% collapse

**Recommendation:** `ValueError` is a broad built-in type carrying 45 different meanings here. Consider defining project-specific exception subclasses, or narrowing handler try-blocks to minimize the catch surface.

### MEDIUM RISK: `TypeError` — 12 raise sites, 7 handlers

**Files:** `app/amqp.py`, `app/base.py`, `backends/base.py`, `backends/consul.py`, `canvas.py`, `fixups/django.py`, `platforms.py`, `schedules.py`, `utils/serialization.py`, `utils/threads.py`, `utils/timer2.py`, `worker/consumer/consumer.py`
**Impact:** `TypeError` is raised at 12 sites across 6 files (amqp.py, base.py, canvas.py, platforms.py, schedules.py, serialization.py), in 8 different functions. 7 handlers (1 re-raise, 1 return, 1 assign default). Information collapse: 86% of semantic information is lost (2.6 bits destroyed).

**Raise sites:**
- `schedules.py:471` raise `TypeError` in `crontab._expand_cronspec` — elif isinstance(cronspec, Iterable) → raise in _expand_cronspec
- `canvas.py:1599` raise `TypeError` in `group.apply_async` — if link is not None → raise in apply_async (`"Cannot add link to group: use a chord`)
- `canvas.py:1601` raise `TypeError` in `group.apply_async` — if link_error is not None → raise in apply_async (`"Cannot add link to group: do that on individual tasks`)
- `platforms.py:653` raise `TypeError` in `Signals.signum` — if not isinstance(name, str) \ → raise in signum (`"signal name must be uppercase string.`)
- `app/amqp.py:333` raise `TypeError` in `AMQP.as_task_v2` — if not isinstance(args, (list, tuple)) → raise in as_task_v2 (`"task args must be a list or tuple`)
- `app/amqp.py:335` raise `TypeError` in `AMQP.as_task_v2` — if not isinstance(kwargs, Mapping) → raise in as_task_v2 (`"task keyword arguments must be a mapping`)
- `app/amqp.py:426` raise `TypeError` in `AMQP.as_task_v1` — if not isinstance(args, (list, tuple)) → raise in as_task_v1 (`"task args must be a list or tuple`)
- `app/amqp.py:428` raise `TypeError` in `AMQP.as_task_v1` — if not isinstance(kwargs, Mapping) → raise in as_task_v1 (`"task keyword arguments must be a mapping`)
- ... and 4 more

**Handlers:**
- `canvas.py:49` — except `TypeError` in `maybe_unroll_group` (handles)
- `fixups/django.py:222` — except `TypeError` in `DjangoWorkerFixup._close_database` (assigns default)
- `backends/base.py:445` — except `TypeError` in `Backend.exception_to_python` (re-raises)
- `backends/consul.py:80` — except `TypeError` in `ConsulBackend.get` (handles)
- `worker/consumer/consumer.py:656` — except `TypeError` in `Consumer.on_task_received` (returns)
- ... and 2 more

**Information theory:** 3.0 bits entropy, 2.6 bits lost, 86% collapse

**Recommendation:** `TypeError` is a broad built-in type carrying 12 different meanings here. Consider defining project-specific exception subclasses, or narrowing handler try-blocks to minimize the catch surface.

### MEDIUM RISK: `RuntimeError` — 18 raise sites, 7 handlers

**Files:** `_state.py`, `app/base.py`, `app/trace.py`, `backends/asynchronous.py`, `backends/base.py`, `backends/redis.py`, `backends/rpc.py`, `canvas.py`, `concurrency/asynpool.py`, `contrib/testing/app.py`, `contrib/testing/worker.py`, `local.py`, `platforms.py`, `result.py`, `schedules.py`, `utils/debug.py`, `utils/log.py`, `utils/threads.py`
**Impact:** `RuntimeError` is raised at 18 sites across 15 files (_state.py, app.py, asynpool.py, base.py, canvas.py, debug.py, local.py, log.py, platforms.py, redis.py, result.py, rpc.py, schedules.py, threads.py, worker.py), in 17 different functions. 7 handlers (1 re-raise, 3 return). Information collapse: 86% of semantic information is lost (3.5 bits destroyed).

**Raise sites:**
- `schedules.py:527` raise `RuntimeError` in `crontab.roll_over` — if datedata.moy == len(months_of_year) → raise in roll_over (`"unable to rollover, time specification is probably invalid`)
- `canvas.py:2097` raise `RuntimeError` in `_chord.freeze` — if node.id in seen → raise in freeze (`"Recursive result parents`)
- `_state.py:110` raise `RuntimeError` in `get_current_app` — in get_current_app (`"USES CURRENT APP`)
- `result.py:38` raise `RuntimeError` in `assert_will_not_block` — if task_join_will_block() → raise in assert_will_not_block
- `platforms.py:413` raise `RuntimeError` in `detached` — if not resource → raise in detached (`"This platform does not support detach.`)
- `local.py:110` raise `RuntimeError` in `Proxy._get_current_object` — in _get_current_object
- `app/base.py:577` raise `RuntimeError` in `Celery._task_from_fun` — if not self.finalized and not self.autofinalize → raise in _task_from_fun (`"Contract breach: app not finalized`)
- `app/base.py:640` raise `RuntimeError` in `Celery.finalize` — if auto and not self.autofinalize → raise in finalize (`"Contract breach: app not finalized`)
- ... and 10 more

**Handlers:**
- `local.py:116` — except `RuntimeError` in `Proxy.__dict__` (re-raises)
- `local.py:122` — except `RuntimeError` in `Proxy.__repr__` (returns)
- `local.py:129` — except `RuntimeError` in `Proxy.__bool__` (returns)
- `local.py:137` — except `RuntimeError` in `Proxy.__dir__` (returns)
- `app/trace.py:314` — except `RuntimeError` in `traceback_clear` (handles)
- ... and 2 more

**Information theory:** 4.1 bits entropy, 3.5 bits lost, 86% collapse

**Recommendation:** `RuntimeError` is a broad built-in type carrying 18 different meanings here. Consider defining project-specific exception subclasses, or narrowing handler try-blocks to minimize the catch surface.

### MEDIUM RISK: `AttributeError` — 9 raise sites, 51 handlers

**Files:** `app/amqp.py`, `app/base.py`, `app/routes.py`, `app/task.py`, `app/trace.py`, `app/utils.py`, `backends/cosmosdbsql.py`, `backends/dynamodb.py`, `backends/rpc.py`, `bin/celery.py`, `canvas.py`, `concurrency/asynpool.py`, `concurrency/prefork.py`, `events/dumper.py`, `local.py`, `platforms.py`, `utils/collections.py`, `utils/debug.py`, `utils/deprecated.py`, `utils/dispatch/signal.py`, `utils/functional.py`, `utils/imports.py`, `utils/log.py`, `utils/term.py`, `utils/threads.py`, `worker/strategy.py`, `worker/worker.py`
**Impact:** `AttributeError` is raised at 9 sites across 6 files (collections.py, deprecated.py, dynamodb.py, local.py, threads.py, utils.py), in 8 different functions. 51 handlers (10 re-raise, 11 return, 14 assign default). Information collapse: 79% of semantic information is lost (2.4 bits destroyed).

**Raise sites:**
- `local.py:117` raise `AttributeError` in `Proxy.__dict__` — in __dict__ (`"__dict__`)
- `app/utils.py:391` raise `AttributeError` in `find_app` — if isinstance(found, ModuleType) → raise in find_app
- `app/utils.py:396` raise `AttributeError` in `find_app` — if isinstance(found, ModuleType) → raise in find_app (`"attribute 'celery' is the celery module not the instance of ...`)
- `backends/dynamodb.py:273` raise `AttributeError` in `DynamoDBBackend._validate_ttl_methods` — if missing_methods → raise in _validate_ttl_methods
- `utils/threads.py:138` raise `AttributeError` in `Local.__getattr__` — in __getattr__
- `utils/threads.py:152` raise `AttributeError` in `Local.__delattr__` — in __delattr__
- `utils/deprecated.py:97` raise `AttributeError` in `_deprecated_property.__set__` — if self.__set is None → raise in __set__ (`"cannot set attribute`)
- `utils/deprecated.py:105` raise `AttributeError` in `_deprecated_property.__delete__` — if self.__del is None → raise in __delete__ (`"cannot delete attribute`)
- ... and 1 more

**Handlers:**
- `canvas.py:333` — except `AttributeError` in `Signature.__init__` (assigns default)
- `canvas.py:1217` — except `AttributeError` in `_chain.prepare_steps` (assigns default)
- `canvas.py:2356` — except `AttributeError` in `_chord._get_app` (assigns default)
- `platforms.py:102` — except `AttributeError` in `isatty` (handles)
- `platforms.py:227` — except `AttributeError` in `Pidfile.write_pid` (handles)
- ... and 46 more

**Information theory:** 3.0 bits entropy, 2.4 bits lost, 79% collapse

**Recommendation:** `AttributeError` is a broad built-in type carrying 9 different meanings here. Consider defining project-specific exception subclasses, or narrowing handler try-blocks to minimize the catch surface.

### MEDIUM RISK: `StopIteration` — 4 raise sites, 12 handlers

**Files:** `app/builtins.py`, `backends/arangodb.py`, `backends/asynchronous.py`, `backends/base.py`, `backends/gcs.py`, `concurrency/asynpool.py`, `utils/functional.py`, `worker/loops.py`
**Impact:** `StopIteration` is raised at 4 sites across 2 files (asynchronous.py, asynpool.py), in 3 different functions. 12 handlers (1 re-raise, 3 return, 5 assign default). Information collapse: 92% of semantic information is lost (1.4 bits destroyed).

**Raise sites:**
- `backends/asynchronous.py:204` raise `StopIteration` in `AsyncBackendMixin.iter_native` — if not results → raise in iter_native
- `concurrency/asynpool.py:921` raise `StopIteration` in `AsynPool._write_job` — if errors > 100 → raise in _write_job
- `concurrency/asynpool.py:937` raise `StopIteration` in `AsynPool._write_job` — if errors > 100 → raise in _write_job
- `concurrency/asynpool.py:971` raise `StopIteration` in `AsynPool._write_ack` — in _write_ack

**Handlers:**
- `app/builtins.py:86` — except `StopIteration` in `unlock_chord` (assigns default)
- `backends/base.py:1184` — except `StopIteration` in `BaseKeyValueStoreBackend.on_chord_part_return` (assigns default)
- `backends/arangodb.py:162` — except `StopIteration` in `ArangoDbBackend.mget` (handles)
- `backends/gcs.py:293` — except `StopIteration` in `GCSBackend.on_chord_part_return` (assigns default)
- `concurrency/asynpool.py:336` — except `StopIteration` in `ResultHandler.on_result_readable` (handles)
- ... and 7 more

**Information theory:** 1.6 bits entropy, 1.4 bits lost, 92% collapse

**Recommendation:** Multiple handlers exist, which may provide adequate discrimination. Verify that each handler's try-block scope only exposes the expected raise sites.

### MEDIUM RISK: `IndexError` — 2 raise sites, 12 handlers

**Files:** `apps/multi.py`, `apps/worker.py`, `bin/control.py`, `canvas.py`, `concurrency/asynpool.py`, `events/cursesmon.py`, `result.py`, `utils/collections.py`, `utils/debug.py`, `worker/consumer/consumer.py`
**Impact:** `IndexError` is raised at 2 sites in 1 different functions. 12 handlers (3 re-raise, 1 return, 2 assign default).

**Raise sites:**
- `utils/collections.py:685` raise `IndexError` in `Evictable._evict1` — if self._evictcount <= self.maxsize → raise in _evict1
- `utils/collections.py:689` raise `IndexError` in `Evictable._evict1` — in _evict1

**Handlers:**
- `canvas.py:390` — except `IndexError` in `Signature.apply_async` (returns)
- `result.py:870` — except `IndexError` in `ResultSet.supports_native_join` (handles)
- `apps/multi.py:344` — except `IndexError` in `MultiParser._update_ns_opts` (re-raises)
- `apps/worker.py:250` — except `IndexError` in `Worker.startup_info` (assigns default)
- `concurrency/asynpool.py:837` — except `IndexError` in `AsynPool.schedule_writes` (handles)
- ... and 7 more

**Recommendation:** Multiple handlers exist, which may provide adequate discrimination. Verify that each handler's try-block scope only exposes the expected raise sites.

### MEDIUM RISK: `self.Empty` — 2 raise sites, 2 handlers

**File:** `utils/collections.py`
**Impact:** `self.Empty` is raised at 2 sites in 2 different functions. 2 handlers (1 re-raise, 1 assign default).

**Raise sites:**
- `utils/collections.py:723` raise `self.Empty` in `Messagebuffer.take` — if default → raise in take
- `utils/collections.py:816` raise `self.Empty` in `BufferMap.take` — if default → raise in take

**Handlers:**
- `utils/collections.py:688` — except `self.Empty` in `Evictable._evict1` (re-raises)
- `utils/collections.py:808` — except `self.Empty` in `BufferMap.take` (assigns default)

**Information theory:** 1.0 bits entropy, 0.5 bits lost, 50% collapse

**Recommendation:** Multiple handlers exist, which may provide adequate discrimination. Verify that each handler's try-block scope only exposes the expected raise sites.

---

## Benchmark Context

| Project | Files | Crossings | Elevated+ | Density |
|---------|-------|-----------|-----------|---------|
| **celery** | **161** | **40** | **4** | **0.25** |
| click | 17 | 11 | 4 | 0.65 |
| requests | 18 | 5 | 2 | 0.28 |
| hypothesis | 103 | 29 | 7 | 0.28 |
| invoke | 47 | 12 | 3 | 0.26 |
| flask | 24 | 6 | 2 | 0.25 |
| tqdm | 31 | 7 | 3 | 0.23 |
| scrapy | 113 | 23 | 8 | 0.20 |
| uvicorn | 40 | 7 | 3 | 0.18 |
| colorama | 7 | 1 | 0 | 0.14 |
| httpx | 23 | 3 | 0 | 0.13 |
| pytest | 71 | 9 | 9 | 0.13 |
| rich | 100 | 5 | 1 | 0.05 |
| fastapi | 47 | 0 | 0 | 0.00 |

celery's crossing density (0.25) is in line with the benchmark average (0.20).

---

## Methodology

Crossing performs static AST analysis on Python source files. It maps every `raise` statement to every `except` handler that could catch it, then identifies **semantic boundary crossings** — places where the same exception type is raised with different meanings in different contexts. No code is executed; no network calls are made; no dependencies are required.

Risk levels:
- **Low:** Single raise site or uniform semantics
- **Medium:** Multiple raise sites in different functions — handler may not distinguish
- **Elevated:** Many divergent raise sites — high chance of incorrect handling
- **High:** Handler collapse — many raise sites, very few handlers, ambiguous behavior

---

*Report generated by [Crossing](https://fridayops.xyz/crossing/) v0.9*  
*Scan performed 2026-02-24*
